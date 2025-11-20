const functions = require("firebase-functions");
const admin = require("firebase-admin");
const OpenAI = require("openai");
const Anthropic = require("@anthropic-ai/sdk");
const moment = require("moment-timezone");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { Logging } = require('@google-cloud/logging');
const { RetrievalQAChain, loadQAStuffChain } = require("langchain/chains");
const { PromptTemplate } = require("@langchain/core/prompts");
const { ChatOpenAI, OpenAIEmbeddings } = require("@langchain/openai");
const { PineconeStore } = require("@langchain/pinecone");
const { Pinecone } = require("@pinecone-database/pinecone");

const logging = new Logging();

// Creates a Google Cloud log event for an API request, DB write, or API response
exports.create_log_event = async function(contactId, locationId, path, type, data) {
  const logData = {
    locationId: locationId,
    contactId: contactId,
    operation: type,
    endpoint: path,
    details: data
  };

  const log = logging.log(locationId);
  const entry = log.entry({}, logData);
  await log.write(entry);
}

// Gets the current date in human readable format
exports.get_human_readable_date = function(date_to_convert=undefined) {
  // 1. Parse the timestamp string to a Date object
  let date = new Date();

  if (date_to_convert) {
    date = date_to_convert;
  }

  // 2. Extract the desired components
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0'); // months are 0-based in JavaScript
  const day = String(date.getDate()).padStart(2, '0');

  let hours = date.getHours();
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');

  // Convert 24-hour time to 12-hour time with AM/PM
  let ampm = hours >= 12 ? 'PM' : 'AM';
  hours = hours % 12;
  hours = hours ? hours : 12; // the hour '0' should be '12'
  hours = String(hours).padStart(2, '0');

  // 3. Format into a human-readable string with AM/PM
  const humanReadableDate = `${year}-${month}-${day} ${hours}:${minutes}:${seconds} ${ampm}`;

  return humanReadableDate;
}

// Gets the current date based on a timezone
exports.get_timezone_date = function(timezone, dateToParse=undefined) {
  let currentDate = new Date();

  if (dateToParse) {
    currentDate = new Date(dateToParse);
  }

  let currentDateMoment = moment(currentDate.toISOString());
  if (timezone) {
    currentDateMoment = currentDateMoment.tz(timezone);
  }
  else {
    currentDateMoment = currentDateMoment.tz("America/Chicago");
  }

  return (new Date(currentDateMoment.toLocaleString().split(" GMT")[0]));  
}

exports.wait_random_time = async function(minSeconds, maxSeconds) {
  const randomTimeInSeconds = Math.random() * (maxSeconds - minSeconds) + minSeconds;
  const randomTimeInMilliseconds = randomTimeInSeconds * 1000;

  return new Promise(resolve => setTimeout(resolve, randomTimeInMilliseconds));
}

// Gets a Firesbase user ID from the user token
exports.get_user_id_from_token = async function(token) {
  try {
    const decodedToken = await admin.auth().verifyIdToken(token);
    return decodedToken.uid;
  } catch (error) {
    return undefined;
  }
}

// Gets a record in the database
exports.get_db_record = async function(db, path) {
  const dataRef = db.doc(path);
  const dataDoc = await dataRef.get();

  if (!dataDoc.exists) {
    return undefined;
  }
  else {
    return dataDoc.data();
  }
}

// Sets a record in the database (creates if not present, otherwise update without clearing non updating fields)
exports.set_db_record = async function(db, path, data) {
  const dataRef = db.doc(path);
  await dataRef.set(data, { merge: true });
}

exports.get_location_company_id = async function(location_id, db) {
  // Retrieve location data from Firestore
  const locationRef = db.collection('LocationTokens').doc(location_id);
  const locationData = (await locationRef.get()).data();
  
  if (!locationData || !locationData.company_id) {
    throw new Error("Location data or company ID for location not found in Firestore"); 
  }

  return locationData.company_id;
}

exports.get_access_token = async function(location_id, db, userType="Location") {
    // Retrieve location data from Firestore
    const locationRef = db.collection('LocationTokens').doc(location_id);
    const locationData = (await locationRef.get()).data();
    
    if (!locationData) {
      throw new Error("Location data not found in Firestore"); 
    }
  
    // Check if current access token is still valid
    const currentTime = new Date();
    const tokenExpiration = new Date(locationData.access_token_expiration);
    
    if (currentTime.getTime() + 8 * 60 * 60 * 1000 < tokenExpiration.getTime()) {
      return locationData.access_token;
    }
  
    // Access token is expired, refresh it
    const url = "https://services.leadconnectorhq.com/oauth/token";

    const payload = new URLSearchParams();
    payload.append('client_id', functions.config().ghl.client_id);
    payload.append('client_secret', functions.config().ghl.client_secret); 
    payload.append('grant_type', 'refresh_token');
    payload.append('refresh_token', locationData.refresh_token);
    payload.append('user_type', userType);    
    
    const headers = {
      "Content-Type": "application/x-www-form-urlencoded"
    };  
  
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: payload    
    });
  
    if (!response.ok) {
      throw new Error(`Failed to refresh access token: ${await response.text()}`);
    }
    
    const newTokenData = await response.json();
    
    const newExpiration = new Date();
    newExpiration.setSeconds(newExpiration.getSeconds() + newTokenData.expires_in);
  
    // Update Firestore with the new token data
    await locationRef.update({
      access_token: newTokenData.access_token,  
      refresh_token: newTokenData.refresh_token,
      access_token_expiration: newExpiration.toISOString() 
    });
  
    return newTokenData.access_token;  
}

exports.check_basic_authorization = function(req) {
  if (Object.hasOwnProperty.call(req.headers, "authorization")) {
    if (!(req.headers.authorization == functions.config().intro_call.authorization_pass)) {
      return "Authorization header incorrect.";
    }
    else {
      return undefined;
    }
  }
  else {
    return "No authorization header.";
  }
}

exports.get_agents_and_tags_for_location = async function(db, companyId, locationId) {
  const agentsSnapshot = await db.collection(`TextualyCompanies/${companyId}/Agents`)
    .where('locations', 'array-contains', locationId)
    .get();

  const agentsSnapshotEmptyLocation = await db.collection(`TextualyCompanies/${companyId}/Agents`)
    .where('locations', '==', [])
    .get();

  let agentsData = [];
  let agentTags = [];

  // Function to process each location documents snapshot
  const processSnapshot = (snapshot) => {
    snapshot.forEach(doc => {
      const docData = doc.data();

      if (docData.enabled) {
        agentsData.push({ id: doc.id, ...docData });
        agentTags = [...new Set([...agentTags, ...(docData.tags || [])])];
      }
    });
  };

  // Process both snapshots
  processSnapshot(agentsSnapshot);
  processSnapshot(agentsSnapshotEmptyLocation);

  // Return combined data
  return { agentsData: agentsData, agentTags: agentTags };
}

exports.get_latest_texts = async function(leadRef, numTexts=10) {
    // Fetch the latest texts_to_fetch messages to determine who sent the last message and when
    const texts_to_fetch = Number.parseInt(numTexts);
    const messagesSnapshot = await leadRef.collection('Messages')
      .orderBy('dateAdded', 'desc')
      .limit(texts_to_fetch)
      .get();
  
    let messagesData = [];
    messagesSnapshot.forEach(doc => {
      messagesData.push(doc.data());
    });
  
    // Reverse the list at the end
    messagesData.reverse();

    return messagesData;
}

exports.create_conversation_string = function(textMessages, timezone) {
    let conversation = "";

    textMessages.forEach(message => {
      // Format the timestamp for readability
      const currentDate = exports.get_timezone_date(timezone, message.dateAdded); 
      const timestamp = exports.get_human_readable_date(currentDate);
      // Indicate the direction of the message
      const direction = message.direction === 'outbound' ? "From us:" : "From lead:";
      // Append each message to the conversation string
      conversation += `${timestamp} ${direction}\n${message.body}\n\n`;
    });

    return conversation;
}

exports.ask_vector_db_question = async function(indexName, openAIApiKey, question, namespace=undefined, metadataFilter=undefined) {
  const pinecone = new Pinecone({
    apiKey: functions.config().pinecone.api_key,
    environment: functions.config().pinecone.environment
  });

  const pineconeIndex = pinecone.index(indexName);

  const storeOptions = { pineconeIndex };
  if (metadataFilter) {
    storeOptions.filter = metadataFilter;
  }
  if (namespace) {
    storeOptions.namespace = namespace;
  }

  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings({ openAIApiKey: openAIApiKey }), storeOptions
  );

  const retriever = vectorStore.asRetriever({ k: 3 });
  const model = new ChatOpenAI({ openAIApiKey: openAIApiKey });

  const template = `Use the following pieces of context to answer the question at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.
  Use three sentences maximum and keep the answer as concise as possible.
  {context}
  Question (including previous messages for more context): {question}
  Helpful Answer:`;

  const QA_CHAIN_PROMPT = new PromptTemplate({
    inputVariables: ["context", "question"],
    template,
  });            

  const chain = new RetrievalQAChain({
    combineDocumentsChain: loadQAStuffChain(model, { prompt: QA_CHAIN_PROMPT }),
    retriever,
    returnSourceDocuments: true,
    inputKey: "question",
  });

  const response = await chain.invoke({question});

  return response;
}

exports.get_location_calendar = async function(db, calendarPath) {
  const calendarDoc = await db.doc(calendarPath).get();

  if (!calendarDoc.exists) {
    return undefined;
  }

  return calendarDoc.data(); 
}

exports.add_or_remove_contact_tag = async function(access_token, contactId, tag, operation) {
  const url = `https://services.leadconnectorhq.com/contacts/${contactId}/tags`;
  const options = {
    method: operation,
    headers: {
      Authorization: `Bearer ${access_token}`,
      Version: '2021-07-28',
      'Content-Type': 'application/json',
      Accept: 'application/json'
    },
    body: JSON.stringify({tags: [tag]})
  };
  
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(error);
    return error;
  }
}

exports.invoke_webhook_with_params = function(url, params) {
  fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(params)
  });
}

exports.get_calendar_availability = async function(access_token, calendarId) {
  const url = `https://services.leadconnectorhq.com/calendars/${calendarId}/free-slots?startDate=0&endDate=2000000000000000`;
  const options = {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${access_token}`,
      Version: '2021-04-15',
      Accept: 'application/json'
    }
  };
  
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(error);
    return undefined;
  }
}

exports.get_contact_appointments = async function(access_token, contactId) {
  const url = `https://services.leadconnectorhq.com/contacts/${contactId}/appointments`;
  const options = {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${access_token}`,
      Version: '2021-07-28',
      Accept: 'application/json'
    }
  };
  
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(error);
    return {statusCode: 400, message: "Couldn't fetch existing appointments."};
  }  
}

exports.cancel_appointment = async function(access_token, calendarId, contactId, locationId, eventId) {
  const url = `https://services.leadconnectorhq.com/calendars/events/appointments/${eventId}`;
  const options = {
    method: 'PUT',
    headers: {
      Authorization: `Bearer ${access_token}`,
      Version: '2021-04-15',
      'Content-Type': 'application/json',
      Accept: 'application/json'
    },
    body: JSON.stringify({calendarId, locationId, contactId, appointmentStatus: "cancelled"})
  };
  
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(error);
    return {statusCode: 400, message: "Couldn't cancel the existing appointment."};
  }  
}

// Cancels an existing appointment to reschedule but still cause the triggers for booking in GHL
exports.cancel_existing_appointment = async function(db, access_token, calendarId, contactId, locationId, companyPath) {
  await exports.add_or_remove_contact_tag(access_token, contactId, "textualy-cancelled", "POST");
  
  // Get existing appointments for the contact
  const contactAppointments = await exports.get_contact_appointments(access_token, contactId);

  if (contactAppointments?.statusCode === 400) {
    return contactAppointments;
  }

  // Check if they are scheduled for the same calendar with any appointment
  const sameCalendarBookings = contactAppointments?.events?.filter((event) => event.calendarId === calendarId) || [];

  // For every event from the same calendar, cancel it because we're about to book another appointment to reschedule
  for (let calendarBooking of sameCalendarBookings) {
    const cancelAppointmentResult = await exports.cancel_appointment(access_token, calendarId, contactId, locationId, calendarBooking.id);

    if (cancelAppointmentResult?.statusCode === 400) {
      return cancelAppointmentResult;
    }
  }

  const didCancel = sameCalendarBookings.length > 0;

  // If an appointment was cancelled, mark the lead as not booked
  if (didCancel) {
    const conversationPath = `${companyPath}/Conversations/${contactId}`;
    await exports.set_db_record(db, conversationPath, {currBooked: false}); 
  }

  return {rescheduled: didCancel};
}

exports.book_appointment = async function(access_token, db, calendarId, locationId, contactId, companyPath, startTime) {
  const cancelAppointmentResult = await exports.cancel_existing_appointment(db, access_token, calendarId, contactId, locationId, companyPath);

  console.log(cancelAppointmentResult);

  if (cancelAppointmentResult?.statusCode === 400) {
    return cancelAppointmentResult;
  }

  await exports.wait_random_time(12, 15);

  const url = 'https://services.leadconnectorhq.com/calendars/events/appointments';
  const options = {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${access_token}`,
      Version: '2021-04-15',
      'Content-Type': 'application/json',
      Accept: 'application/json'
    },
    body: JSON.stringify({calendarId, locationId, contactId, startTime, toNotify: true})
  };
  
  try {
    const response = await fetch(url, options);
    const data = await response.json();
  

    // Add the tag that signals that Textualy booked the lead
    // await exports.add_or_remove_contact_tag(access_token, contactId, "textualy-booked", "POST");

    // Update the conversation in the DB to say that Textualy booked the lead
    const conversationPath = `${companyPath}/Conversations/${contactId}`;
    await exports.set_db_record(db, conversationPath, {booked: true, currBooked: true, rescheduled: cancelAppointmentResult.rescheduled || false});

    return data;
  } catch (error) {
    console.error(error);
    return {statusCode: 400, message: "Couldn't book the appointment at the requested time."};
  }
}
  
exports.send_sms = async function(apiKey, contactId, message) {
    // Sends an SMS through the GHL API.
    const url = "https://services.leadconnectorhq.com/conversations/messages";

    const payload = {
      type: "SMS",
      contactId: contactId,
      message: message
    };
    const headers = {
      "Authorization": `Bearer ${apiKey}`,
      "Version": "2021-04-15",
      "Content-Type": "application/json",
      "Accept": "application/json"
    };
  
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload)
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Failed to send SMS:", error);
      throw error;
    }
}

exports.get_ghl_contact_data = async function(access_token, contactId, retries=0) {
  // Queries GHL to get the contact information for the contact in the current conversation
  const url = `https://services.leadconnectorhq.com/contacts/${contactId}`;
  const headers = {
      'Accept': 'application/json',
      'Authorization': `Bearer ${access_token}`,
      'Version': '2021-07-28'
  };

  const response = await fetch(url, { method: 'GET', headers });
  const leadData = (await response.json())["contact"];   

  if (!leadData && retries < 3) {
    await new Promise((resolve) => setTimeout(resolve, 10000)); // 10 seconds delay
    return exports.get_ghl_contact_data(access_token, contactId, retries + 1);
  }

  return leadData;
}