const functions = require("firebase-functions");
const admin = require("firebase-admin");

const helpers = require("./helpers");
const textualy_graph = require("./textualy_graph");

const cors = require('cors')({origin: true});

/*
Example request body:

{
  "companyId": "kVE8ut9G45Dp2J70p0FZ",
  "conversationId": "Mon Feb 26 2024 11:57:19 AM"
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
        const requiredProperties = ["companyId", "conversationId"];
        
        // Check for missing properties
        for (const property of requiredProperties) {
            if (!reqBody.hasOwnProperty(property)) {
            const bodyJson = {"success": false, "reason": `Request body missing required parameter: ${String(property)}`};
            return res.status(403).send(bodyJson);
            }
        }

        try {
            const token = req.headers.authorization?.split('Bearer ')[1];
            if (!token) {
                return res.status(403).send('No token supplied on the request.');
            }

            // Gets the firestore instance
            const firestore = admin.firestore();

            // Get the user and their associated company ID from the token
            const userId = await helpers.get_user_id_from_token(token);
            const userDoc = await firestore.doc(`users/${userId}`).get();

            if (!userDoc.exists) {
                return res.status(401).send('User data does not exist.');
            }

            const userData = userDoc.data();
            const userCompanyId = userData.ghlCompanyId;

            // If the user's company ID doesn't match the requested ID, deny the whole request.
            if (userCompanyId !== reqBody["companyId"]) {
                return res.status(403).send('User company ID does not match request company ID.');
            }

            const companyPath = `TextualyCompanies/${userCompanyId}`;
            const conversationPath = `${companyPath}/EmulatorConversations/${reqBody["conversationId"]}`;
            const conversationDoc = await firestore.doc(conversationPath).get();
            
            if (!conversationDoc.exists) {
                return res.status(401).send('Conversation does not exist.');
            }

            const conversationData = conversationDoc.data();
            const locationId = conversationData.location;
            const agentId = conversationData.agent;

            // Gets the AI response to the latest text in the conversation
            const timezone = "America/Chicago";
            let response = await textualy_graph.get_langchain_ai_response(firestore, companyPath, agentId, locationId, "EMULATOR_CONTACT", conversationPath, timezone, simulate=true);
            const answer = response.answer;

            if (!response.success) {
                const bodyJson = {"success": false, "reason": response.reason};
                await helpers.create_log_event("Emulator", locationId, "function/emulator_conversation", "APIResponse", bodyJson);
                return res.status(403).send(bodyJson);
            }
        
            const bodyJson = {"success": true, answer};
            await helpers.create_log_event("Emulator", locationId, "function/emulator_conversation", "APIResponse", bodyJson);             
            return res.status(200).send(bodyJson);
          }
          catch (error) {
            const bodyJson = {"error": true, "success": false, "reason": `Internal error - ${error.message}: ${error.stack}`};
            await helpers.create_log_event("Emulator", "NA", "function/emulator_conversation", "APIResponse", bodyJson);          
            return res.status(500).send(bodyJson);
          }
    });    
};
