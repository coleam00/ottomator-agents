# Slack Bot Listener Test

This test script verifies the basic functionality of the Slack Bot Listener by:
1. Starting a listener for your configured test channels
2. Listening for messages
3. Simulating the agent callback process
4. Printing detailed instructions on how to test the integration

## Prerequisites

1. Create a `.env` file based on the provided `env.example` file
2. Configure your Slack API credentials and test channel(s) in the `.env` file

## Running the Test

1. Make sure your virtual environment is activated (if using one)
2. Run the test script:
```bash
python test_slackbot_listener.py
```

## Test Instructions

When you run the test, you'll see instructions on the screen:

1. The test will generate a unique MESSAGE_ID
2. Send any message to the configured test channel
3. The test will receive your message and simulate an agent processing it
4. In a real integration, the agent would respond with a message that starts with `[MESSAGE_ID]-...`
5. The test will run for 60 seconds or until interrupted with Ctrl+C

## Expected Output

If everything is working correctly:
- You'll see log messages confirming the bot received your message
- The test will show what would happen in a real integration scenario
- You'll see details about what message the agent would send back

## Troubleshooting

If the test fails:
1. Check your `.env` configuration
2. Verify your Slack API tokens are valid
3. Ensure the bot has been added to the test channel
4. Check the log messages for any errors 