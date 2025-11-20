const functions = require("firebase-functions");
const admin = require("firebase-admin");
const { SystemMessage, HumanMessage, AIMessage, ToolMessage } = require("@langchain/core/messages");
const { ToolNode } = require("@langchain/langgraph/prebuilt");
const { ChatOpenAI } = require("@langchain/openai");
const { StateGraphArgs } = require("@langchain/langgraph");
const { END } = require("@langchain/langgraph");
const { START, StateGraph } = require("@langchain/langgraph");

const helpers = require("./helpers");
const textualy_tools = require("./textualy_tools");
 
// LangGraph state - all context needed for the conversation
const agentState = {
    messages: {
        value: (x, y) => x.concat(y),
        default: () => [],
    },
    textualy: {
        // Overwrite if a new Textualy object is provided
        value: (x, y) => (y ? y : x),
        default: () => undefined
    },
    locationId: {
        value: (x, y) => (y ? y : x),
        default: () => undefined
    },
    contactId: {
        value: (x, y) => (y ? y : x),
        default: () => undefined
    },
    companyPath: {
        value: (x, y) => (y ? y : x),
        default: () => undefined
    },
    conversationPath: {
        value: (x, y) => (y ? y : x),
        default: () => undefined
    },
    simulate: {
        value: (x, y) => (y ? y : x),
        default: () => undefined
    }
};

// Define the function that determines whether to continue or not
const shouldContinue = (state) => {
    const { messages } = state;
    const lastMessage = messages[messages.length - 1];
    // If there is no function call, then we finish
    if (!lastMessage?.tool_calls?.length) {
      return END;
    }
    else {
      return "tools";
    }
};

// Define the function that handles all tool calls
const toolNode = async (state, config) => {
    const tools = textualy_tools.tools;
    const { messages, locationId, contactId, companyPath, conversationPath, simulate } = state;
    const lastMessage = messages[messages.length - 1];

    // Loop over every tool call and make the call + create the ToolMessage for the call results
    const outputs = await Promise.all(
        lastMessage.tool_calls?.map(async (call) => {
            const tool = tools.find((tool) => tool.name === call.name);

            // Tool not found
            if (tool === undefined) {
                throw new Error(`Tool "${call.name}" not found.`);
            }

            // Invoke the tool with the arguments the AI requested
            // as well as the extra context needed like location ID, contact ID, etc.
            const output = await tool.invoke(
                { ...call.args, locationId: locationId, contactId: contactId, companyPath: companyPath, simulate: simulate },
                config
            );

            // Crafts the new database entry for the new tool call messasge
            const currDate = new Date();
            const currDateStr = currDate.toISOString();
            const messageData = {
                body: typeof output === "string" ? output : JSON.stringify(output),
                dateAdded: currDateStr,
                direction: "outbound",
                userId: "TextualyAI",
                userName: "Textualy AI",
                tool_name: tool.name,
                tool_call_id: call.id
            }

            // Add the tool message to the conversation in the DB
            const firestore = admin.firestore();
            const messageDataPath = `${conversationPath}/Messages/${currDateStr}`;
            await helpers.set_db_record(firestore, messageDataPath, messageData);
            const functionName = simulate ? "emulator_conversation" : "ghl_conversation";
            await helpers.create_log_event("Emulator", locationId, `function/${functionName}`, "DBWrite", {dbPath: messageDataPath, ...messageData});            

            return new ToolMessage({
                name: tool.name,
                content: typeof output === "string" ? output : JSON.stringify(output),
                tool_call_id: call.id
            });
        }) ?? []
    );    

    return { messages: outputs }
}
  
// Define the function that calls the model
const callModel = async (state, config) => {
    const { messages, textualy, conversationPath, locationId, simulate } = state;  

    // Invoke the LLM with the conversation history and LangGraph config
    const response = await textualy.invoke(messages, config);

    // Crafts the new database entry for the new text messasge from the AI
    const currDate = new Date();
    const currDateStr = currDate.toISOString();
    let messageData = {
        body: response.content,
        dateAdded: currDateStr,
        direction: "outbound",
        userId: "TextualyAI",
        userName: "Textualy AI",
        showUser: response.content.length > 0
    }

    if (response?.tool_calls?.length) {
      messageData.tool_calls = response.tool_calls;
    }

    // Stores the AI's response in the DB
    const firestore = admin.firestore();
    const messageDataPath = `${conversationPath}/Messages/${currDateStr}`;
    await helpers.set_db_record(firestore, messageDataPath, messageData);
    const functionName = simulate ? "emulator_conversation" : "ghl_conversation";
    await helpers.create_log_event("Emulator", locationId, `function/${functionName}`, "DBWrite", {dbPath: messageDataPath, ...messageData});
    
    // Return an object, because this will get added to the existing list
    return { messages: [response] };
};

// Defines the graph for the whole lead nurturing process
const workflow = new StateGraph({ channels: agentState })
  // Define the two nodes it will cycle between
  .addNode("agent", callModel)
  // Note the "action" and "final" nodes are identical
  .addNode("tools", toolNode)
  .addNode("final", toolNode)
  // Set the entrypoint as as the node that invokes the LLM
  .addEdge(START, "agent")
  // We now add a conditional edge
  .addConditionalEdges(
    // First, define the start node which is the agent
    "agent",
    // Next, pass in the function that will determine which node is called next.
    shouldContinue,
  )
  // Now add a normal edge from `tools` to `agent`.
  .addEdge("tools", "agent")
  .addEdge("final", END);

// Formats an array of messages into the format needed for the LLM
// (human messages, AI messages, tool messages, etc.)
exports.format_conversation_for_llm = function(prompt, textMessages) {
    let conversation = [];
    conversation.push(new SystemMessage(prompt));

    textMessages.forEach(message => {
      conversation.push(
        message.tool_call_id ? new ToolMessage({
            name: message.tool_name,
            tool_call_id: message.tool_call_id,
            content: message.body
        })
        :
        (
            message.direction === "inbound"
                ? new HumanMessage(message.body) 
                : new AIMessage(content=message.body, additional_kwargs=(message.tool_calls ? {
                  tool_calls: message.tool_calls.map((tool_call) => ({
                    ...tool_call,
                    type: "function",
                    "function": { name: tool_call.name, arguments: JSON.stringify(tool_call.args) }
                  }))
                } : {}))
        )
      );
    });

    return conversation;
}

// Function to invoke the LangGraph executable to get the next conversation message from the AI
// and invoke any necessary tools to book on the calendar, add a tag, get calendar availability, etc.
exports.get_llm_response = async function(agent, conversation, locationId, contactId, companyPath, conversationPath, simulate) {
    const apiKey = agent.openAIAPIKey;

    // Creates the LLM instance (just OpenAI right now but this could easily be extended to include other models)
    const model = new ChatOpenAI({
        model: agent.model,
        apiKey
    });
    const textualy = model.bindTools(textualy_tools.tools);

    // Compiles the LangGraph graph for execution
    const app = workflow.compile();

    // Defines all the inputs for the graph including extra context needed
    // for the conversation like the location ID and contact ID
    const inputs = { messages: conversation, textualy: textualy, locationId, contactId, companyPath, conversationPath, simulate };

    // Streams the output from the LangGraph execution and returns the final response from the AI
    // after any potential tool calls
    let finalMessage = "";
    for await (const output of await app.stream(inputs, { streamMode: "values" })) {
        const lastMessage = output.messages[output.messages.length - 1];
        finalMessage = lastMessage;
    }

    return finalMessage.content;
}

// Function to add all the dynamic actions to the prompt
// This is how you can have different calendars from different GHL locations,
// instructions on when specifically to book a lead on the calendar,
// different tags to add to leads in different situations, etc.
exports.build_actions = async function(db, companyPath, locationId, agent) {
    let actions = "";
  
    // Get the data for each action and append the instructions to the action string
    // actionType can be one of: "Text Calendar Availability" | "Book Appointment" | "Cancel Appointment" | "Add Tag" | "Remove Tag" | "Invoke Webhook"
    for (let actionId of agent.actions) {
      const actionPath = `${companyPath}/Actions/${actionId}`;
      const actionDoc = await db.doc(actionPath).get();
    
      if (!actionDoc.exists) {
        continue;
      }
    
      const { actionType, actionParameter, trigger } = actionDoc.data();

      let actionParamText = `'${actionParameter}'`;

      // Extra context needed if it is a calendar based action
      // since the LLM needs the calendar ID and the calendar name
      if (actionParameter === "Set the calendar for this action in each location individually") {
        const calendarPath = `${companyPath}/Locations/${locationId}/ActionCalendars/${actionId}`;
        const locationCalendar = await helpers.get_location_calendar(db, calendarPath);
        actionParamText = `with calendar ID: '${locationCalendar.calendarId} and calendar name: ${locationCalendar.calendarName})`;
      }

      actions += `\n${trigger} - ${actionType} ${actionParamText}`;
    }
  
    return actions;
  }

// Main function to get the next message in the GHL conversation from the LLM
// This has RAG for FAQs, dynamic prompts and actions, all the bells and whistles!
exports.get_langchain_ai_response = async function(db, companyPath, agentId, locationId, contactId, conversationPath, timezone, simulate=false) {
    // Gets the access token for the location and refreshes it if necessary
    const access_token = await helpers.get_access_token(locationId, db);
  
    const agentPath = `${companyPath}/Agents/${agentId}`;
    const agentDoc = await db.doc(agentPath).get();
  
    if (!agentDoc.exists) {
        return {success: false, reason: "Agent does not exist."};
    }
  
    const agent = agentDoc.data();  
  
    // Get the location context
    const locationPath = `${companyPath}/Locations/${locationId}`;
    const locationDoc = await db.doc(locationPath).get();
  
    if (!locationDoc.exists) {
        return {success: false, reason: "Location does not exist."};
    }
  
    const locationData = locationDoc.data();  
    const locationContext = locationData.context;
  
    const promptDoc = await db.doc(`${companyPath}/Prompts/${agent.prompt}`).get();
    const conversationRef = db.doc(conversationPath);
    const textMessages = await helpers.get_latest_texts(conversationRef, functions.config().ghl.texts_to_fetch);
  
    // There need to be messages and a valid prompt to continue
    if (textMessages.length === 0 || !promptDoc || !promptDoc.data().prompt) {
        return {success: false, reason: "Prompt or texts invalid for this request."};
    }
  
    const conversationStr = helpers.create_conversation_string(
      textMessages.length > 2 ? textMessages.slice(-2) : [...textMessages],
      timezone
    );
  
    // Checks the knowledgebase with RAG for anything pertaining to the text and adds it to the prompt if it finds something
    let faqContext = "";
    try {
      const response = await helpers.ask_vector_db_question(
        functions.config().pinecone.index,
        agent.openAIAPIKey,
        conversationStr,
        locationId
      );
      
      if (response.sourceDocuments) {
        faqContext = response.sourceDocuments.reduce((accumulator, currentDocument) => {
          return accumulator + currentDocument.pageContent + '\n\n';
        }, '');
      }
    } catch (error) {
      console.log(error);
    }
  
    // Include any actions in the prompt if it's the initial prompt and the agent has actions tied to it
    let actions = "";
    if (agent.actions.length) {
      actions = await exports.build_actions(db, companyPath, locationId, agent);
    }
  
    // Build up the prompt by replacing the placeholders with the timestamp and conversation
    const currDateTime = helpers.get_human_readable_date(helpers.get_timezone_date(timezone));  
    let prompt = promptDoc.data().prompt;
    const originalPrompt = prompt;
    
    // Add on the context fetched from the regular DB and vector DB
    prompt += faqContext ? `\n\nFAQ for more context: \n${faqContext}` : "\n\n";
    prompt += locationContext ? `More context for the location: \n${locationContext}\n\n` : ""; 
  
    // Only include actions instructions if this isn't the second prompt after performing an action
    prompt += actions ? `List of specific triggers that require you to prepend your response with the ID of an action to take (only choose zero or one): \n${actions}\nIf the lead asks for availablility, take that action even if you think you know the availability already.` : "";
  
    // Adds on the context given for every message, including date information for days such as today, tomorrow, the next day, etc.
    prompt += `\n\nHere is some information for you on dates: ${helpers.generate_date_context(timezone)}`;
  
    // Add the current timestamp to the prompt so the AI knows what time it is
    prompt += `\n\nThe current time in the timezone of the lead is: ${currDateTime}. Your output will be sent directly to the lead as the next text message.`;

    // Create the conversation - each message includes the timestamp, from us or them, and the message
    const conversation = exports.format_conversation_for_llm(prompt, textMessages);
    
    // Get the next text message from the AI with the above prompt
    const answer = await exports.get_llm_response(agent, conversation, locationId, contactId, companyPath, conversationPath, simulate);
    
    return {success: true, prompt: originalPrompt, answer, agent};
  }