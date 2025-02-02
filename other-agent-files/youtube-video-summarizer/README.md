# YouTube Video Summarizer Agent

Author: [Mike Russell](https://n8n.io/creators/mikerussell/)

This n8n-powered agent is a conversational AI assistant that creates comprehensive summaries of YouTube videos from Cole Medin's channel. It can process both video IDs and full YouTube links, providing detailed summaries and engaging in follow-up discussions about the video content.

## Features

- Processes YouTube video IDs and full URLs
- Retrieves and analyzes video captions
- Generates detailed video summaries
- Supports conversational follow-up questions
- Maintains context for natural dialogue
- Extracts key points and insights

## Technical Details

- **Platform**: n8n workflow automation
- **Model**: OpenAI GPT
- **Integration**: YouTube Data API v3
- **Authentication**: YouTube OAuth2
- **Response Format**: Conversational text

## Important Note

⚠️ **YouTube API Authentication Required**: This agent requires YouTube OAuth2 credentials from your own YouTube account to access video captions. You must set up YouTube API credentials for your account before using this agent if you are running it yourself, as it cannot access captions using credentials from other accounts.

## Use Cases

1. **Video Content Review**
   - Quick understanding of video content
   - Extract main topics and key points
   - Review video highlights without watching

2. **Research and Learning**
   - Study video content efficiently
   - Extract specific information
   - Follow up with clarifying questions

3. **Content Analysis**
   - Identify main themes
   - Track topic progression
   - Understand key takeaways

4. **Time-Saving**
   - Get video summaries instantly
   - Focus on relevant sections
   - Decide if full video watch is needed

## Conversation Examples

You can interact with the agent in various ways:

1. Basic video summary:
   ```
   "Summarize this video: https://youtube.com/watch?v=..."
   ```

2. Follow-up questions:
   ```
   "What were the main points about [specific topic]?"
   "What are your thoughts on this video?"
   ```

3. Specific inquiries:
   ```
   "What tools or technologies were mentioned?"
   "What were the key recommendations?"
   ```
