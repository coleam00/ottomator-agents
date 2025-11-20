const functions = require("firebase-functions");
const admin = require("firebase-admin");

const locationKnowledgeBase = require("./location_knowledge_base");
const GHLConversationLangChain = require("./ghl_conversation_langchain");
const EmulatorConversationLangChain = require("./emulator_conversation_langchain");

if (functions.config().firestore.GOOGLE_PRIVATE_KEY) {
    const firebaseJson = {
        projectId: functions.config().firestore.GOOGLE_PROJECT_ID,
        clientEmail: functions.config().firestore.GOOGLE_CLIENT_EMAIL,
        privateKey: functions.config().firestore.GOOGLE_PRIVATE_KEY?.replaceAll(/\\n/g, '\n')
    }

    admin.initializeApp({ credential: admin.credential.cert(firebaseJson) });
}
else {
    admin.initializeApp();
}

exports.updateLocationKnowledgeBase = functions.https.onRequest(async (req, res) => {
    locationKnowledgeBase.handler(req, res);   
});

exports.GHLConversationLangChain = functions.runWith({
    timeoutSeconds: 200,
    memory: '256MB'
  }).https.onRequest(async (req, res) => {
    GHLConversationLangChain.handler(req, res);   
});

exports.EmulatorConversationLangChain = functions.runWith({
    timeoutSeconds: 200,
    memory: '256MB'
  }).https.onRequest(async (req, res) => {
    EmulatorConversationLangChain.handler(req, res);   
});