"use client";

import React from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

export interface VoiceflowButton {
  name: string;
  request: {
    type: string;
    payload: {
      label: string;
      actions: any[];
    };
  };
}

export interface VoiceflowTrace {
  type: string;
  payload: {
    message?: string;
    slate?: {
      content: Array<{
        children: Array<{
          text: string;
        }>;
      }>;
    };
    buttons?: VoiceflowButton[];
  };
}

export interface VoiceflowAgentProps {
  data: VoiceflowTrace[];
  handleSendMessage: (message?: string, messageData?: any) => Promise<void>;
}

export const VoiceflowAgent = React.memo(function VoiceflowAgent({ data, handleSendMessage }: VoiceflowAgentProps) {
  if (!data || data.length === 0) {
    return (
      <p className="text-muted-foreground">
        Error getting response from agent...
      </p>
    );
  }

  const handleButtonClick = async (button: VoiceflowButton) => {
    const messageData = {
      text: button.name,
      data: {
        request: {
          type: button.request.type,
          payload: {
            label: button.request.payload.label
          }
        }
      }
    };
    await handleSendMessage(button.name, `\`\`\`json${JSON.stringify(messageData)}`);
  };

  const renderTrace = (trace: VoiceflowTrace) => {
    switch (trace.type) {
      case 'path':
        return null;
      case 'text':
        const message = trace.payload.message || 
          trace.payload.slate?.content[0]?.children[0]?.text || 
          'Empty message';
        return (
          <p className="text-foreground">
            {message}
          </p>
        );
      case 'choice':
        if (!trace.payload.buttons?.length) return null;
        return (
          <div className="flex flex-wrap gap-2 mt-2">
            {trace.payload.buttons.map((button, index) => (
              <Button
                key={index}
                variant="secondary"
                onClick={() => handleButtonClick(button)}
                className="text-sm"
              >
                {button.name}
              </Button>
            ))}
          </div>
        );
      case 'knowledgeBase':
        return (
          <p className="text-muted-foreground italic">
            Searched Knowledgebase
          </p>
        );
      default:
        return (
          <p className="text-muted-foreground italic">
            Taking custom action: {trace.type}
          </p>
        );
    }
  };

  return (
    <Card className="p-4 my-4 space-y-2 bg-card">
      {data.map((trace, index) => (
        <React.Fragment key={index}>
          {renderTrace(trace)}
        </React.Fragment>
      ))}
    </Card>
  );
});
