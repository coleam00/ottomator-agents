'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { SendIcon, MenuIcon, HomeIcon, MessageCircleIcon, SettingsIcon, LogOutIcon } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs'
import { Session, AuthChangeEvent } from '@supabase/supabase-js'
import { Auth } from '@supabase/auth-ui-react'
import { ThemeSupa } from '@supabase/auth-ui-shared'
import { useRouter } from 'next/navigation'

type Message = {
  text: string
  sender: 'user' | 'bot'
}

const exampleConversations = [
  { title: "Getting Started", id: "getting-started" },
  { title: "Task Planning", id: "task-planning" },
  { title: "Problem Solving", id: "problem-solving" },
  { title: "Creative Writing", id: "creative-writing" },
]

export default function AIAgentChatbot() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [user, setUser] = useState<any>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const supabase = createClientComponentClient()
  const router = useRouter()

  useEffect(() => {
    const getUser = async () => {
      const { data: { user } } = await supabase.auth.getUser()
      setUser(user)
    }
    getUser()

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event: AuthChangeEvent, session: Session | null) => {
      setUser(session?.user ?? null)
    })

    return () => subscription.unsubscribe()
  }, [supabase.auth])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  const handleSignOut = async () => {
    await supabase.auth.signOut()
    router.refresh()
  }

  const handleSendMessage = async () => {
    if (inputMessage.trim() === '') return

    const newUserMessage: Message = { text: inputMessage, sender: 'user' }
    setMessages(prevMessages => [...prevMessages, newUserMessage])
    setInputMessage('')
    setIsTyping(true)

    try {
      const response = await fetch('[YOUR n8n WORKFLOW WEBHOOK URL]', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer [YOUR AUTH BEARER TOKEN]'
        },
        body: JSON.stringify({
          sessionId: user?.id,
          chatInput: inputMessage
        })
      })

      if (!response.ok) {
        throw new Error('Network response was not ok')
      }

      const data = await response.json()
      const botResponse: Message = { text: data.output, sender: 'bot' }
      setMessages(prevMessages => [...prevMessages, botResponse])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage: Message = { text: 'Sorry, I encountered an error. Please try again.', sender: 'bot' }
      setMessages(prevMessages => [...prevMessages, errorMessage])
    } finally {
      setIsTyping(false)
    }
  }

  if (!user) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900 text-gray-100">
        <div className="w-full max-w-md p-8 space-y-8 bg-gray-800 rounded-lg shadow-md">
          <h2 className="text-2xl font-bold text-center">Sign In to AI Agent Chatbot</h2>
          <Auth
            supabaseClient={supabase}
            appearance={{ theme: ThemeSupa }}
            theme="dark"
            providers={['google', 'github']}
            redirectTo={`${window.location.origin}/auth/callback`}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-gray-800 transform ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0`}>
        <div className="flex flex-col h-full">
          <div className="flex items-center justify-center h-16 bg-gray-900">
            <h2 className="text-xl font-bold">Conversations</h2>
          </div>
          <ScrollArea className="flex-1">
            {exampleConversations.map((conv) => (
              <button
                key={conv.id}
                className="w-full text-left px-4 py-2 hover:bg-gray-700 focus:outline-none focus:bg-gray-700"
              >
                {conv.title}
              </button>
            ))}
          </ScrollArea>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-col flex-1">
        {/* Navigation header */}
        <header className="flex items-center justify-between px-4 h-16 bg-gray-800 shadow-md">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="lg:hidden"
            >
              <MenuIcon className="h-6 w-6" />
              <span className="sr-only">Toggle sidebar</span>
            </Button>
            <Button variant="ghost" size="icon">
              <HomeIcon className="h-6 w-6" />
              <span className="sr-only">Home</span>
            </Button>
            <Button variant="ghost" size="icon">
              <MessageCircleIcon className="h-6 w-6" />
              <span className="sr-only">Messages</span>
            </Button>
            <Button variant="ghost" size="icon">
              <SettingsIcon className="h-6 w-6" />
              <span className="sr-only">Settings</span>
            </Button>
          </div>
          <h1 className="text-xl font-bold">AI Agent Chatbot</h1>
          <Button variant="ghost" size="icon" onClick={handleSignOut}>
            <LogOutIcon className="h-6 w-6" />
            <span className="sr-only">Sign out</span>
          </Button>
        </header>

        {/* Chat area */}
        <ScrollArea className="flex-grow p-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`mb-4 ${
                message.sender === 'user' ? 'text-right' : 'text-left'
              }`}
            >
              <span
                className={`inline-block p-2 rounded-lg ${
                  message.sender === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-100'
                }`}
              >
                {message.sender === 'bot' ? (
                  <ReactMarkdown>{message.text}</ReactMarkdown>
                ) : (
                  message.text
                )}
              </span>
            </div>
          ))}
          {isTyping && (
            <div className="text-left mb-4">
              <span className="inline-block p-2 rounded-lg bg-gray-700 text-gray-100">
                Typing...
              </span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </ScrollArea>

        {/* Input area */}
        <div className="p-4 border-t border-gray-700">
          <form
            onSubmit={(e) => {
              e.preventDefault()
              handleSendMessage()
            }}
            className="flex space-x-2"
          >
            <Input
              type="text"
              placeholder="Type your message..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              className="flex-1 bg-gray-700 text-gray-100 border-gray-600 focus:border-blue-500"
            />
            <Button type="submit" className="bg-blue-600 hover:bg-blue-700">
              <SendIcon className="h-4 w-4" />
              <span className="sr-only">Send</span>
            </Button>
          </form>
        </div>
      </div>
    </div>
  )
}