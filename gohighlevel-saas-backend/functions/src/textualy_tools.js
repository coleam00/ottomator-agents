
const { DynamicStructuredTool } = require("@langchain/core/tools");
const admin = require("firebase-admin");
const { z } = require("zod");

const helpers = require("./helpers");

const AddTagSchema = z.object({
    locationId: z.string().optional().describe("The location ID of the lead"),
    contactId: z.string().optional().describe("The contact ID of the lead"),
    simulate: z.boolean().optional().describe("Whether or not this is a simulated run"),
    tag: z.string().describe("The tag to add to the lead")
});

const RemoveTagSchema = z.object({
    locationId: z.string().optional().describe("The location ID of the lead"),
    contactId: z.string().optional().describe("The contact ID of the lead"),
    simulate: z.boolean().optional().describe("Whether or not this is a simulated run"),
    tag: z.string().describe("The tag to remove from the lead")
});

const InvokeWebhookSchema = z.object({
    locationId: z.string().optional().describe("The location ID of the lead"),
    contactId: z.string().optional().describe("The contact ID of the lead"),
    simulate: z.boolean().optional().describe("Whether or not this is a simulated run"),    
    url: z.string().describe("The webhook URL to invoke")
});

const CalendarAvailabilitySchema = z.object({
    locationId: z.string().optional().describe("The location ID of the lead"),
    calendarId: z.string().describe("The ID of the calendar to get the availability from")
});

const CancelAppointmentSchema = z.object({
    locationId: z.string().optional().describe("The location ID of the lead"),
    contactId: z.string().optional().describe("The contact ID of the lead"),
    simulate: z.boolean().optional().describe("Whether or not this is a simulated run"),
    companyPath: z.string().optional().describe("The path in the DB to the company data"),
    calendarId: z.string().describe("The ID of the calendar to get the availability from"),
    calendarName: z.string().describe("The name of the calendar")
});

const BookAppointmentSchema = z.object({
    locationId: z.string().optional().describe("The location ID of the lead"),
    contactId: z.string().optional().describe("The contact ID of the lead"),
    simulate: z.boolean().optional().describe("Whether or not this is a simulated run"),
    companyPath: z.string().optional().describe("The path in the DB to the company data"),
    calendarId: z.string().describe("The ID of the calendar to get the availability from"),
    calendarName: z.string().describe("The name of the calendar"),
    bookingTime: z.string().describe("The time to book the appointment in the format 2024-02-25T11:00:00")
});

const addTagTool = new DynamicStructuredTool({
    name: "add_tag",
    description: "Call to add a tag to the lead",
    schema: AddTagSchema,
    func: async ({locationId, contactId, simulate, tag}) => {
        if (!simulate) {
            const db = admin.firestore();        
            const access_token = await helpers.get_access_token(locationId, db);
            return (await helpers.add_or_remove_contact_tag(access_token, contactId, tag, "POST"));            
        }
        else {
            console.log(`Tag "${tag}" added to lead.`);
            return `Tag "${tag}" added to lead.`;
        }
    }
});

const removeTagTool = new DynamicStructuredTool({
    name: "remove_tag",
    description: "Call to remove a tag from the lead",
    schema: RemoveTagSchema,
    func: async ({locationId, contactId, simulate, tag}) => {
        if (!simulate) {
            const db = admin.firestore();        
            const access_token = await helpers.get_access_token(locationId, db);
            return (await helpers.add_or_remove_contact_tag(access_token, contactId, tag, "DELETE"));            
        }
        else {
            console.log(`Tag "${tag}" removed from lead.`);
            return `Tag "${tag}" removed from lead.`;
        }
    }
});

const invokeWebhookTool = new DynamicStructuredTool({
    name: "invoke_webhook",
    description: "Call to invoke a webhook",
    schema: InvokeWebhookSchema,
    func: async ({locationId, contactId, simulate, url}) => {
        if (!simulate) {
            helpers.invoke_webhook_with_params(actionParameter, {locationId, contactId});
        }
        else {
            console.log(`URL "${url}" would have been invoked.`);
        }

        return `URL "${url}" was invoked.`;
    }
});

const getCalendarAvailabilityTool = new DynamicStructuredTool({
    name: "get_calendar_availability",
    description: "Call to get availability from a calendar",
    schema: CalendarAvailabilitySchema,
    func: async ({locationId, calendarId}) => {
        const db = admin.firestore();        
        const access_token = await helpers.get_access_token(locationId, db);
        return (await helpers.get_calendar_availability(access_token, calendarId));
    }
});

const cancelAppointmentTool = new DynamicStructuredTool({
    name: "cancel_appointment",
    description: "Call to cancel an appointment",
    schema: CancelAppointmentSchema,
    func: async ({locationId, contactId, simulate, companyPath, calendarId, calendarName}) => {
        const db = admin.firestore();
        if (!simulate) {   
            const access_token = await helpers.get_access_token(locationId, db);
            return (await helpers.cancel_existing_appointment(db, access_token, calendarId, contactId, locationId, companyPath));
        }
        else {
            return `Appointment has have been cancelled on the calendar "${calendarName}"`;
        }
    }
});

const bookAppointmentTool = new DynamicStructuredTool({
    name: "book_appointment",
    description: "Call to book an appointment",
    schema: BookAppointmentSchema,
    func: async ({locationId, contactId, simulate, companyPath, calendarId, calendarName, bookingTime}) => {
        const db = admin.firestore();
        if (!simulate) {   
            const access_token = await helpers.get_access_token(locationId, db);
            return (await helpers.book_appointment(access_token, db, calendarId, locationId, contactId, companyPath, bookingTime));
        }
        else {
            return `Appointment has been booked for "${bookingTime}" on the calendar "${calendarName}"`;
        }
    }
});

exports.tools = [addTagTool, removeTagTool, invokeWebhookTool, getCalendarAvailabilityTool, cancelAppointmentTool, bookAppointmentTool];