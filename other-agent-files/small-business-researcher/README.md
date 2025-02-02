# Reddit Small Business Researcher

Author: [Zubair Trabzada](https://www.youtube.com/@AI-GPTWorkshop)

An n8n-powered agent that researches small business ideas by analyzing Reddit discussions from r/smallbusiness. It helps validate business concepts by finding relevant experiences, challenges, and insights from real business owners.

## Features

- Searches r/smallbusiness for relevant discussions
- Filters high-quality posts based on:
  - Upvotes (minimum of 2)
  - Post content length
  - Recent posts (within last 180 days)
- Uses AI to analyze and summarize findings
- Provides actionable insights and potential challenges

## How It Works

1. Takes your business idea as input
2. Searches Reddit for relevant discussions
3. Filters and processes the most relevant posts
4. Analyzes content using AI to extract valuable insights
5. Provides a summarized report with key findings

## Example Usage

### Input
```json
{
    "query": "I want to start a mobile car detailing business"
}
```

### Example Response
```
Based on analysis of recent r/smallbusiness discussions:

Key Insights:
1. Market Demand
   - High demand in suburban areas
   - Popular with busy professionals
   - Seasonal fluctuations reported

2. Startup Costs
   - Initial equipment: $2,000-5,000
   - Vehicle requirements
   - Insurance considerations

3. Common Challenges
   - Weather dependencies
   - Client scheduling
   - Water source access
   - Competition from established services

4. Success Factors
   - Quality equipment investment
   - Professional image
   - Reliable booking system
   - Clear service packages

5. Revenue Potential
   - Average job: $150-300
   - Monthly potential: $5,000-10,000
   - Repeat customer opportunities

Recommendations:
- Start with basic services, expand based on demand
- Invest in good scheduling software
- Consider offering packages for regular clients
- Build relationships with local businesses
- Focus on photo documentation for marketing

Recent Success Story:
"Started 8 months ago, now fully booked with 70% repeat customers. Key was focusing on quality over speed and building a strong Instagram presence."
```

## Usage Notes

1. Provide specific business ideas for better results
2. The agent focuses on recent discussions (last 6 months)
3. Results are based on real experiences shared on Reddit
4. Analysis includes both positive and negative insights
5. Recommendations are data-driven from actual business owners

## Credentials Required

- Reddit API credentials (for accessing r/smallbusiness)
- OpenAI API key (for content analysis)

## Contributing

This agent is part of the oTTomator agents collection. For contributions or issues, please refer to the main repository guidelines.
