const functions = require("firebase-functions");
const admin = require("firebase-admin");

const helpers = require("./helpers");
const textualy_graph = require("./textualy_graph");

const cors = require('cors')({origin: true});

/*
Example request body:

{
  "type": "OutboundMessage",
  "locationId": "kVE8ut9G45Dp2J70p0FZ",
  "attachments": [],
  "body": "Hey Sheliah! This is Mark from Bi-County Chiropractic and Rehab! I saw you filled out the facebook form for our program just now\nFeel free to reply STOP to unsubscribe!",
  "contactId": "py3Xl8kbtLsnoFkaU2El",
  "contentType": "text/plain",
  "conversationId": "Bo2X3qpiJlrQty5qKXKb",
  "dateAdded": "2023-10-14T17:57:23.000Z",
  "direction": "outbound",
  "messageType": "SMS"
}
*/

exports.handler = function(req, res) {
    cors(req, res, async () => {
        if (req.method === 'OPTIONS') {
            res.set('Access-Control-Allow-Origin', '*');
            res.set('Access-Control-Allow-Methods', 'GET, POST');
            res.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
            res.set('Access-Control-Max-Age', '3600');
            return res.status(204).send('');
        }
    
        const reqBody = req.body;

        // Stops the request if the required fields are not present
        // List of required properties
        const requiredProperties = [
            "type",
            "locationId",
            "body",
            "contentType",
            "conversationId",
            "dateAdded",
            "direction",
            "messageType",
            "contactId"
        ];
        
        // Check for missing properties
        for (const property of requiredProperties) {
            if (!reqBody.hasOwnProperty(property)) {
            console.error(`Request body missing required parameter: ${String(property)}`);
            return res.status(403).send("Request body missing required parameter");
            }
        }

        // Fetches all metadata for the conversation - lead data, location ID, company ID, and access token
        const { locationId, contactId, direction, body } = reqBody;

        try {    
            // Gets the firestore instance
            const firestore = admin.firestore();

            await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIRequest", reqBody);            

            const companyId = await helpers.get_location_company_id(locationId, firestore);
            const access_token = await helpers.get_access_token(locationId, firestore);
            const leadData = await helpers.get_ghl_contact_data(access_token, contactId);
            
            // All of Textualy going forward
            // First determine the path to the conversation in the DB based on the company, location, and contact
            const companyPath = `TextualyCompanies/${companyId}`;
            const conversationPath = `${companyPath}/Conversations/${contactId}`;
            
            // Second, create the conversation document if it doesn't exist already
            let currDate = new Date();
            let currDateStr = currDate.toISOString();

            // Next, determine the agent to use based on tags, location, etc.
            const leadTags = leadData["tags"];
            if (!leadTags) {
                const bodyJson = {"success": false, "reason": 'No tags for this lead.'};
                await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                     
                return res.status(401).send(bodyJson);
            }

            // Get all the agents for the location as well as a compiled list of tags across all of those locations
            const { agentsData, agentTags } = await helpers.get_agents_and_tags_for_location(firestore, companyId, locationId);

            if (!agentsData.length) {
                const bodyJson = {"success": false, "reason": 'No enabled agents found for this lead based on the location.'};
                await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                     
                return res.status(401).send(bodyJson);
            }

            // Assign to the agent ID to ID of the first agent that has a tag match with a tag of the GHL contact
            let agentId = "";
            for (let tag of leadTags) {
                if (agentTags.includes(tag)) {
                    // Assign the first agent that has this tag
                    agentId = agentsData.filter((agent) => agent.tags.includes(tag))[0]?.id || "";
                    break;
                }
            }

            if (!agentId) {
                const bodyJson = {"success": false, "reason": 'No agent found for this lead based on the location and tags.'};
                await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                     
                return res.status(401).send(bodyJson);
            }

            let conversationDoc = await firestore.doc(conversationPath).get();
        
            if (!conversationDoc.exists) {
                try {
                    const conversationData = { 
                        dateStarted: currDateStr,
                        dateUpdatedISO: (new Date()).toISOString(),
                        contactEmail: leadData["email"] || "",
                        contactFullName: `${leadData["firstName"] || ""} ${leadData["lastName"] || ""}`,
                        contactPhoneNumber: leadData["phone"] || "",
                        lastAgentId: "",
                        locationId: locationId,
                        agents: []
                    };                 
                    await helpers.set_db_record(firestore, conversationPath, conversationData);
                    await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "DBWrite", {dbPath: conversationPath, ...conversationData});                        
                }
                catch (error) {
                    const bodyJson = {"error": true, "success": false, "reason": `Failed to create conversation in DB: ${error.message}: ${error.stack}`};
                    await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                          
                    return res.status(500).send(bodyJson);
                }     
            }               

            // Save database entry for the new text messasge from the lead if direction is inbound or from the AI if the direction is outbound
            let messageData = {};
            if (direction == "inbound") {
                messageData = {
                    body: body,
                    dateAdded: reqBody.dateAdded,
                    direction: "inbound",
                    userId: contactId,
                    userName: `${leadData["firstName"]} ${leadData["lastName"]}`
                };
            }
            else {
                messageData = {
                    body: body,
                    dateAdded: reqBody.dateAdded,
                    direction: "outbound",
                    userId: "TextualyAI",
                    userName: "Textualy AI",
                    agentId
                };                
            }

            const conversationRef = firestore.doc(conversationPath);
            let textMessages = await helpers.get_latest_texts(conversationRef, 1);
            const messageDataPath = `${conversationPath}/Messages/${currDateStr}`;
            const messageRef = firestore.doc(messageDataPath);
            const messageDoc = await messageRef.get();            
            try {
                if (messageDoc.exists) {
                    const bodyJson = {"success": false, "reason": "Request already processed"};
                    await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                       
                    return res.status(422).send(bodyJson);
                }

                if (!textMessages[0] || textMessages[0]?.body !== body || direction === "inbound" || textMessages[0].direction !== direction) {
                    await helpers.set_db_record(firestore, messageDataPath, messageData);
                    await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "DBWrite", {dbPath: messageDataPath, ...messageData});
                }                       
            }
            catch (error) {
                const bodyJson = {"error": true, "success": false, "reason": `Failed to save AI message to DB: ${error.message}: ${error.stack}`};
                await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                        
                return res.status(500).send(bodyJson);
            }   
            
            // Don't continue if it's an outbound message
            if (direction !== "inbound") {
                const bodyJson = {"success": true};
                await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                       
                return res.status(200).send(bodyJson);  
            }
            
            // Don't continue with the conversation if the lead replied stop
            if (body.toLowerCase() == "stop") {
                const bodyJson = {"success": false, "reason": "Lead replied STOP"};
                await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                       
                return res.status(403).send(bodyJson);                    
            }

            // Get the timestamp of the text that triggered this response from Textualy
            // If any text from them or us is sent after this, there will be another execution
            // of this function so we want this one to end
            textMessages = await helpers.get_latest_texts(conversationRef, 1);
            const originalTimestamp = textMessages[0].dateAdded;

            // Wait a random time in case the lead double texts
            // and to make the response more human-like
            await helpers.wait_random_time(50, 80);
            
            // Only continue if there hasn't been a newer text message
            // Also GHL can sometimes send messages out of order, so if the last message
            // is actually after the current one being processed, continue anyway
            textMessages = await helpers.get_latest_texts(conversationRef, 1);

            if (textMessages.length > 0 && textMessages[0].dateAdded !== originalTimestamp) {              
                const bodyJson = {"success": false, "reason": "New text message sent since the previous one was processed here."};
                await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                     
                return res.status(409).send(bodyJson);
            }            

            // After determining the agent to use, get the AI response to the latest text in the conversation
            const timezone = leadData["timezone"];
            let response = await textualy_graph.get_langchain_ai_response(firestore, companyPath, agentId, locationId, "EMULATOR_CONTACT", conversationPath, timezone, simulate=true);

            if (!response.success) {
                const bodyJson = {"error": true, "success": false, "reason": response.reason};
                await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                       
                return res.status(403).send(bodyJson);
            }
            
            // Only continue if there hasn't been a newer text message
            textMessages = await helpers.get_latest_texts(conversationRef, 1);
            if (textMessages.length > 0 && textMessages[0].dateAdded !== originalTimestamp) {               
                const bodyJson = {"success": false, "reason": "New text message sent since the previous one was processed here."};
                await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                     
                return res.status(409).send(bodyJson);
            }            

            // Only send the SMS message if an agent still applies to this lead
            // Doesn't have to be the same agent but at least some agent still needs to apply
            const leadDataNew = await helpers.get_ghl_contact_data(access_token, contactId);
            const leadTagsNew = leadDataNew["tags"];

            let newAgentId = "";
            for (let tag of leadTagsNew) {
                if (agentTags.includes(tag)) {
                    // Assign the first agent that has this tag
                    newAgentId = agentsData.filter((agent) => agent.tags.includes(tag))[0]?.id || "";
                    break;
                }
            }

            if (!newAgentId) {
                // Delete the message since it isn't getting sent
                const resMessageDataPath = `${conversationPath}/Messages/${textMessages[0].dateAdded}`;
                const resMessageRef = firestore.doc(resMessageDataPath); 
                await resMessageRef.delete();                 
                const bodyJson = {"success": false, "reason": 'Agent no longer applies to this lead - not sending message.'};
                await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                    
                return res.status(401).send(bodyJson);
            }

            // Update the last agent used for the lead and the list of agents used for the lead
            conversationDoc = await firestore.doc(conversationPath).get();
            const conversationData = conversationDoc.data();

            try {
                if (!(conversationData.agents || []).includes(agentId) || conversationData.lastAgentId !== agentId) {
                    const conversationDBData = {
                        lastAgentId: agentId, 
                        agents: [...new Set([...(conversationData.agents || []), agentId])],
                        dateUpdatedISO: (new Date()).toISOString(),
                        replied: true
                    };
                    await helpers.set_db_record(firestore, conversationPath, conversationDBData);
                    await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "DBWrite", {dbPath: conversationPath, ...conversationDBData});                        
                }      
                else {
                    const conversationDBData = {
                        dateUpdatedISO: (new Date()).toISOString(),
                        replied: true
                    };
                    await helpers.set_db_record(firestore, conversationPath, conversationDBData);  
                    await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "DBWrite", {dbPath: conversationPath, ...conversationDBData});                                             
                }              
            } catch (error) {
                const bodyJson = {"error": true, "success": false, "reason": `Failed to update the conversation in DB: ${error.message}: ${error.stack}`};
                await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);                    
                return res.status(500).send(bodyJson);
            }

            // Send the AI reply to the lead through the GHL API
            await helpers.send_sms(access_token, contactId, answer);
        
            const bodyJson = {"success": true};
            await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);            
            return res.status(200).send(bodyJson);
          }
          catch (error) {
            let bodyJson = {"error": true, "success": false, "reason": `Internal error - ${error.message}: ${error.stack}`};

            if (error.message.includes("Location data or company ID for location not found in Firestore")) {
                bodyJson.error = false;
            }

            await helpers.create_log_event(contactId, locationId, "function/ghl_conversation", "APIResponse", bodyJson);
            return res.status(500).send(bodyJson);
          }
    });    
};
