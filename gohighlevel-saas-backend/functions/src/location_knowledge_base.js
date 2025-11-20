const functions = require("firebase-functions");
const admin = require("firebase-admin");
const { Document } = require("@langchain/core/documents");
const { OpenAIEmbeddings } = require("@langchain/openai");
const { PineconeStore } = require("@langchain/pinecone");
const { Pinecone } = require("@pinecone-database/pinecone");

const helpers = require("./helpers");

const cors = require('cors')({origin: true});

exports.handler = function(req, res) {
    cors(req, res, async () => {
        if (req.method === 'OPTIONS') {
            res.set('Access-Control-Allow-Origin', '*');
            res.set('Access-Control-Allow-Methods', 'GET, POST');
            res.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
            res.set('Access-Control-Max-Age', '3600');
            return res.status(204).send('');
            }
    
        // Check for POST request
        if (req.method !== "POST") {
            return res.status(400).send('Please send a POST request');
        }
    
        const reqBody = req.body;

        // Stops the request if the required fields are not present
        // List of required properties
        const requiredProperties = ["locationId", "oldFAQ", "newFAQ"];
        
        // Check for missing properties
        for (const property of requiredProperties) {
            if (!reqBody.hasOwnProperty(property)) {
                const bodyJson = {"success": false, "reason": `Request body missing required parameter: ${String(property)}`};
                return res.status(403).send(bodyJson);
            }
        }

        // Deconstruct the request body to get everything needed for the knowledgebase update
        const { locationId, oldFAQ, newFAQ } = reqBody;

        try {
            const token = req.headers.authorization?.split('Bearer ')[1];
            if (!token) {
                return res.status(403).send('No token supplied on the request.');
            }  

            await helpers.create_log_event("NA", locationId, "function/location_knowledge_base", "APIRequest", reqBody);

            // Can't have more than 100 FAQs
            if (newFAQ.length > 100) {
                const bodyJson = {"success": false, "reason": "Can't have more than 100 FAQs for a location."};
                await helpers.create_log_event("NA", locationId, "function/location_knowledge_base", "APIResponse", bodyJson);
                return res.status(403).send(bodyJson);
            }

            // Gets the firestore instance
            const firestore = admin.firestore();

            // Get the user and their associated company ID from the token
            const userId = await helpers.get_user_id_from_token(token);
            const userDoc = await firestore.doc(`users/${userId}`).get();

            if (!userDoc.exists) {
                const bodyJson = {"success": false, "reason": "User data does not exist."};
                await helpers.create_log_event("NA", locationId, "function/location_knowledge_base", "APIResponse", bodyJson);                
                return res.status(401).send(bodyJson);
            }

            const userData = userDoc.data();
            const userCompanyId = userData.ghlCompanyId;

            // Get the company ID based on the location ID
            const companyId = await helpers.get_location_company_id(locationId, firestore);

            // If the user's company ID doesn't match the requested ID, deny the whole request.
            if (userCompanyId !== companyId) {
                const bodyJson = {"success": false, "reason": "User company ID does not match the company ID for the requested location."};
                await helpers.create_log_event("NA", locationId, "function/location_knowledge_base", "APIResponse", bodyJson);                  
                return res.status(403).send(bodyJson);
            }

            // Get the open AI api key from the first agent for the company
            const companyPath = `TextualyCompanies/${userCompanyId}`;
            const agentsPath = `${companyPath}/Agents`;
            const agentDocs = await firestore.collection(agentsPath).get();

            if (agentDocs.empty) {
                const bodyJson = {"success": false, "reason": "You must create an agent before updating location configuration."};
                await helpers.create_log_event("NA", locationId, "function/location_knowledge_base", "APIResponse", bodyJson);                    
                return res.status(403).send(bodyJson);
            }

            const firstAgentAPIKey = agentDocs.docs[0].data().openAIAPIKey;

            if (!firstAgentAPIKey) {
                const bodyJson = {"success": false, "reason": "Your agents need Open AI API keys before you can configure the knowledgebase for locations."};
                await helpers.create_log_event("NA", locationId, "function/location_knowledge_base", "APIResponse", bodyJson);                     
                return res.status(403).send(bodyJson);
            }
        
            // Initialize Pinecone
            const pinecone = new Pinecone({
                apiKey: functions.config().pinecone.api_key,
                environment: functions.config().pinecone.environment
            });
        
            const pineconeIndex = pinecone.index(functions.config().pinecone.index);

            // Get the vector store from Pinecone with the namespace for the location
            const vectorStore = await PineconeStore.fromExistingIndex(
                new OpenAIEmbeddings({ openAIApiKey: firstAgentAPIKey }), { pineconeIndex, namespace: locationId });
        
            const newFAQQuestions = newFAQ.map((faq) => faq.question);
            const oldFAQQuestions = oldFAQ.map((faq) => faq.question);

            // Delete vectors for FAQs that were deleted
            const deletedFAQQuestions = oldFAQQuestions.filter((question) => !newFAQQuestions.includes(question));
            if (deletedFAQQuestions.length) {
                await vectorStore.delete({
                    ids: deletedFAQQuestions,
                    namespace: locationId
                });
            }

            // Upsert the new FAQs
            const newFAQFiltered = newFAQ
                .filter(faq => 
                    !oldFAQQuestions.includes(faq.question) ||
                    oldFAQ.find(f => f.question === faq.question).answer !== faq.answer
                )

            const newFAQDocs = newFAQFiltered
                .map(faq => new Document({
                    metadata: {
                        question: faq.question,
                        answer: faq.answer
                    },
                    pageContent: `Question: ${faq.question}\nAnswer: ${faq.answer}`
                }));

            const newFAQIds = newFAQFiltered.map((faq) => faq.question);

            let newFAQVectorIds = [];
            if (newFAQDocs.length) {
                newFAQVectorIds = await vectorStore.addDocuments(newFAQDocs, {ids: newFAQIds, namespace: locationId});
            }
        
            const bodyJson = {"success": true, "newFAQVectorIds": newFAQVectorIds};
            await helpers.create_log_event("NA", locationId, "function/location_knowledge_base", "APIResponse", bodyJson);
            return res.status(200).send(bodyJson);
          }
          catch (error) {
            const bodyJson = {"success": false, "reason": `Internal error - ${error.message}: ${error.stack}`};
            await helpers.create_log_event("NA", locationId, "function/location_knowledge_base", "APIResponse", bodyJson);
            return res.status(500).send(bodyJson);
          }
    });    
};
