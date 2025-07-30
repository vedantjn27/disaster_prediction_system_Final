"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Volume2, VolumeX, MessageSquare, Send, Globe, Play, Pause } from "lucide-react"

interface VoiceChatProps {
  backendConnected: boolean
  backendUrl: string
}

export default function VoiceChat({ backendConnected, backendUrl }: VoiceChatProps) {
  const [selectedLanguage, setSelectedLanguage] = useState("en")
  const [textInput, setTextInput] = useState("")
  const [chatHistory, setChatHistory] = useState([
    {
      type: "bot",
      message: "Hello! I'm your AI disaster response assistant. How can I help you today?",
      timestamp: new Date().toISOString(),
      language: "en",
      id: "initial",
    },
  ])
  const [isProcessing, setIsProcessing] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null)
  const [playingMessageId, setPlayingMessageId] = useState<string | null>(null)
  const [ttsError, setTtsError] = useState<string | null>(null)

  const languages = [
    { code: "en", name: "English", flag: "🇺🇸" },
    { code: "hi", name: "हिंदी", flag: "🇮🇳" },
    { code: "kn", name: "ಕನ್ನಡ", flag: "🇮🇳" },
    { code: "te", name: "తెలుగు", flag: "🇮🇳" },
    { code: "ta", name: "தமிழ்", flag: "🇮🇳" },
    { code: "bn", name: "বাংলা", flag: "🇧🇩" },
  ]

  // Cleanup audio on component unmount
  useEffect(() => {
    return () => {
      if (currentAudio) {
        currentAudio.pause()
        currentAudio.src = ""
      }
    }
  }, [currentAudio])

  const handleTextToSpeech = async (text: string, messageId?: string) => {
    if (!text.trim()) return

    console.log("Starting TTS for:", text.substring(0, 50) + "...")
    console.log("Platform:", navigator.platform)
    console.log("User Agent:", navigator.userAgent)
    setTtsError(null)

    // Stop any currently playing audio
    if (currentAudio) {
      currentAudio.pause()
      currentAudio.currentTime = 0
      setCurrentAudio(null)
    }

    setIsSpeaking(true)
    setPlayingMessageId(messageId || null)

    try {
      if (backendConnected) {
        console.log("Calling TTS endpoint...")

        // Call your API endpoint
        const response = await fetch(`${backendUrl}/api/text-to-speech`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: text,
            language: selectedLanguage,
            speed: 150,
          }),
        })

        console.log("TTS Response status:", response.status)

        if (response.ok) {
          const data = await response.json()
          console.log("TTS Response data:", data)

          if (data.success && data.audio_url) {
            // Fetch the actual audio file
            const audioResponse = await fetch(`${backendUrl}${data.audio_url}`)

            if (audioResponse.ok) {
              const audioBlob = await audioResponse.blob()
              console.log("Audio blob size:", audioBlob.size, "bytes")
              console.log("Audio blob type:", audioBlob.type)

              if (audioBlob.size === 0) {
                throw new Error("Received empty audio file")
              }

              // Check if the blob is actually audio
              if (!audioBlob.type.startsWith("audio/")) {
                console.warn("Received non-audio content:", audioBlob.type)
                // Try to determine if it's actually audio data despite wrong MIME type
                if (audioBlob.size < 100) {
                  throw new Error(`Received invalid audio data. Size: ${audioBlob.size} bytes, Type: ${audioBlob.type}`)
                }
              }

              // Create audio URL with explicit type hint for better browser compatibility
              const audioUrl = URL.createObjectURL(new Blob([audioBlob], { type: "audio/wav" }))
              console.log("Created audio URL:", audioUrl)

              const audio = new Audio()
              setCurrentAudio(audio)

              // Enhanced audio settings for better Windows compatibility
              audio.preload = "auto"
              audio.volume = 1.0
              audio.crossOrigin = "anonymous"

              // Set up event listeners with better error handling
              audio.onloadstart = () => console.log("Audio loading started")
              audio.oncanplay = () => console.log("Audio can play")
              audio.oncanplaythrough = () => console.log("Audio can play through")
              audio.onplay = () => console.log("Audio started playing")
              audio.onended = () => {
                console.log("Audio playback ended")
                setIsSpeaking(false)
                setPlayingMessageId(null)
                URL.revokeObjectURL(audioUrl)
                setCurrentAudio(null)
              }
              audio.onerror = (e) => {
                console.error("Audio playback error:", e)
                console.error("Audio error details:", audio.error)
                console.error("Audio src:", audio.src)
                console.error("Audio readyState:", audio.readyState)
                console.error("Audio networkState:", audio.networkState)

                let errorMsg = "Audio playback failed"
                if (audio.error) {
                  errorMsg = `Audio error: ${audio.error.code} - ${getAudioErrorMessage(audio.error.code)}`

                  // Specific handling for format not supported error
                  if (audio.error.code === 4) {
                    errorMsg += ". Try using a different browser or check if audio codecs are installed."
                  }
                }

                setIsSpeaking(false)
                setPlayingMessageId(null)
                setTtsError(errorMsg)
                URL.revokeObjectURL(audioUrl)
                setCurrentAudio(null)
              }
              audio.onpause = () => {
                console.log("Audio paused")
                setIsSpeaking(false)
                setPlayingMessageId(null)
              }

              // Set the source and try to play
              audio.src = audioUrl

              try {
                // Wait for the audio to load properly
                await new Promise((resolve, reject) => {
                  const timeout = setTimeout(() => {
                    reject(new Error("Audio loading timeout"))
                  }, 5000)

                  audio.oncanplay = () => {
                    clearTimeout(timeout)
                    resolve(true)
                  }

                  audio.onerror = () => {
                    clearTimeout(timeout)
                    reject(new Error("Audio failed to load"))
                  }
                })

                await audio.play()
                console.log("Audio play() succeeded")
              } catch (playError) {
                console.error("Audio play() failed:", playError);
              
                if (playError instanceof Error) {
                  setTtsError(`Could not play audio: ${playError.message}. Try clicking to enable audio.`);
                } else {
                  setTtsError("Could not play audio. Try clicking to enable audio.");
                }
              
                setIsSpeaking(false);
                setPlayingMessageId(null);
                URL.revokeObjectURL(audioUrl);
              }
            } else {
              const errorText = await audioResponse.text()
              console.error("Failed to fetch audio file:", audioResponse.status, errorText)
              throw new Error(`Failed to fetch audio file: ${audioResponse.status}`)
            }
          } else {
            throw new Error(data.message || "TTS generation failed")
          }
        }
      } else {
        // Enhanced demo mode for Windows
        console.log("Demo mode: simulating TTS on Windows")

        // Use Web Speech API if available (Windows 10+ with Edge/Chrome)
        if ("speechSynthesis" in window) {
          const utterance = new SpeechSynthesisUtterance(text)
          utterance.lang = selectedLanguage === "hi" ? "hi-IN" : "en-US"
          utterance.onend = () => {
            setIsSpeaking(false)
            setPlayingMessageId(null)
          }
          utterance.onerror = (e) => {
            console.error("Speech synthesis error:", e)
            setIsSpeaking(false)
            setPlayingMessageId(null)
          }

          window.speechSynthesis.speak(utterance)
          console.log("Using Windows Speech Synthesis API")
        } else {
          await new Promise((resolve) => setTimeout(resolve, 2000))
          setIsSpeaking(false)
          setPlayingMessageId(null)
        }
      }
    } catch (error) {
      console.error("TTS failed:", error);
    
      if (error instanceof Error) {
        setTtsError(error.message || "TTS failed");
      } else {
        setTtsError("TTS failed");
      }
    
      setIsSpeaking(false);
      setPlayingMessageId(null);
    }
  }

  // Add helper function for audio error messages
  const getAudioErrorMessage = (errorCode: number) => {
    switch (errorCode) {
      case 1:
        return "MEDIA_ERR_ABORTED - Audio playback was aborted"
      case 2:
        return "MEDIA_ERR_NETWORK - Network error occurred"
      case 3:
        return "MEDIA_ERR_DECODE - Audio decoding error"
      case 4:
        return "MEDIA_ERR_SRC_NOT_SUPPORTED - Audio format not supported"
      default:
        return "Unknown audio error"
    }
  }

  const stopSpeaking = () => {
    console.log("Stopping speech...")
    if (currentAudio) {
      currentAudio.pause()
      currentAudio.currentTime = 0
      setCurrentAudio(null)
    }
    setIsSpeaking(false)
    setPlayingMessageId(null)
  }

  const handleTextSubmit = async () => {
    if (!textInput.trim()) return

    const userMessage = textInput
    const messageId = Date.now().toString()
    setTextInput("")
    setIsProcessing(true)
    setTtsError(null)

    setChatHistory((prev) => [
      ...prev,
      {
        type: "user",
        message: userMessage,
        timestamp: new Date().toISOString(),
        language: selectedLanguage,
        id: `user-${messageId}`,
      },
    ])

    try {
      let botResponse = ""

      if (backendConnected) {
        // Use your chat endpoint (assuming you have one)
        const response = await fetch(`${backendUrl}/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: userMessage,
            language: selectedLanguage,
          }),
        })

        if (response.ok) {
          const data = await response.json()
          botResponse = data.response || data.message || "I received your message."
        } else {
          throw new Error("Chat request failed")
        }
      } else {
        // Simulate AI response
        await new Promise((resolve) => setTimeout(resolve, 1500))

        const responses = [
          "I understand your concern. Here are some safety guidelines for your situation.",
          "Based on current weather conditions, I recommend taking the following precautions.",
          "This is important information. Let me provide you with the most relevant safety advice.",
          "Thank you for reaching out. Here's what you should know about disaster preparedness.",
        ]
        botResponse = responses[Math.floor(Math.random() * responses.length)]
      }

      const botMessageId = `bot-${messageId}`
      setChatHistory((prev) => [
        ...prev,
        {
          type: "bot",
          message: botResponse,
          timestamp: new Date().toISOString(),
          language: selectedLanguage,
          hasAudio: true,
          id: botMessageId,
        },
      ])

      // Automatically speak the bot response after a short delay
      if (botResponse) {
        setTimeout(() => handleTextToSpeech(botResponse, botMessageId), 500)
      }
    } catch (error) {
      console.error("Chat failed:", error)
      const errorMessage = "I'm sorry, I'm having trouble processing your request right now. Please try again."
      const botMessageId = `bot-error-${messageId}`

      setChatHistory((prev) => [
        ...prev,
        {
          type: "bot",
          message: errorMessage,
          timestamp: new Date().toISOString(),
          language: selectedLanguage,
          hasAudio: true,
          id: botMessageId,
        },
      ])
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="bg-gradient-to-r from-purple-50 to-pink-50">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <MessageSquare className="h-6 w-6 text-purple-600" />
            <span>AI Voice Assistant</span>
            <Badge variant="outline" className="bg-purple-100 text-purple-700">
              {backendConnected ? "Multi-language AI" : "Demo Mode"}
            </Badge>
          </CardTitle>
          <CardDescription>
            {backendConnected
              ? "Chat with AI and hear responses using text-to-speech in multiple languages"
              : "Voice assistant running in demo mode (Backend not connected)"}
          </CardDescription>
        </CardHeader>
      </Card>

      {!backendConnected && (
        <Alert className="border-yellow-200 bg-yellow-50">
          <MessageSquare className="h-4 w-4 text-yellow-600" />
          <AlertDescription className="text-yellow-700">
            Backend not connected. Voice features are simulated for demonstration purposes.
          </AlertDescription>
        </Alert>
      )}

      {ttsError && (
        <Alert className="border-red-200 bg-red-50">
          <VolumeX className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-700">
            <strong>TTS Error:</strong> {ttsError}
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chat Interface */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Conversation</span>
                <div className="flex items-center space-x-2">
                  <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
                    <SelectTrigger className="w-40">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {languages.map((lang) => (
                        <SelectItem key={lang.code} value={lang.code}>
                          {lang.flag} {lang.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {isSpeaking && (
                    <Button variant="outline" size="sm" onClick={stopSpeaking}>
                      <Pause className="h-3 w-3 mr-1" />
                      Stop
                    </Button>
                  )}
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {/* Chat History */}
              <div className="h-96 overflow-y-auto mb-4 space-y-4 p-4 bg-gray-50 rounded-lg">
                {chatHistory.map((message, index) => (
                  <div key={index} className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                        message.type === "user" ? "bg-blue-500 text-white" : "bg-white border shadow-sm"
                      }`}
                    >
                      <p className="text-sm">{message.message}</p>
                      <div className="flex items-center justify-between mt-2">
                        <span className="text-xs opacity-70">{new Date(message.timestamp).toLocaleTimeString()}</span>
                        {message.type === "bot" &&  (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleTextToSpeech(message.message, message.id)}
                            disabled={isSpeaking && playingMessageId !== message.id}
                            className="h-6 w-6 p-0"
                          >
                            {playingMessageId === message.id && isSpeaking ? (
                              <VolumeX className="h-3 w-3 text-green-600" />
                            ) : (
                              <Volume2 className="h-3 w-3" />
                            )}
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                {isProcessing && (
                  <div className="flex justify-start">
                    <div className="bg-white border shadow-sm px-4 py-2 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                        <span className="text-sm">AI is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Text Input */}
              <div className="flex space-x-2">
                <Input
                  placeholder="Type your message..."
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  onKeyPress={(e) => e.key === "Enter" && handleTextSubmit()}
                  disabled={isProcessing}
                />
                <Button onClick={handleTextSubmit} disabled={isProcessing || !textInput.trim()}>
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Voice Controls */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Volume2 className="h-5 w-5 text-green-600" />
                <span>Text-to-Speech</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={
                  isSpeaking
                    ? stopSpeaking
                    : () => {
                        const lastBotMessage = chatHistory.filter((m) => m.type === "bot").pop()
                        if (lastBotMessage) {
                          handleTextToSpeech(lastBotMessage.message, lastBotMessage.id)
                        }
                      }
                }
                disabled={isProcessing}
                className={`w-full h-20 text-lg ${
                  isSpeaking ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600"
                }`}
              >
                {isSpeaking ? (
                  <>
                    <Pause className="h-6 w-6 mr-2" />
                    Stop Speaking
                  </>
                ) : (
                  <>
                    <Play className="h-6 w-6 mr-2" />
                    {backendConnected ? "Speak Last Response" : "Simulate TTS"}
                  </>
                )}
              </Button>

              <div className="text-center text-sm text-gray-600">
                {isSpeaking && (
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span>Speaking...</span>
                  </div>
                )}
                {!isSpeaking && !isProcessing && <span>Click the speaker icons to hear messages</span>}
              </div>

              {/* Debug Info */}
              {backendConnected && (
                <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
                  <div>Language: {selectedLanguage}</div>
                  <div>Status: {isSpeaking ? "Speaking" : "Ready"}</div>
                  <div>Backend: Connected</div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Globe className="h-5 w-5 text-blue-600" />
                <span>Language Support</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {languages.map((lang) => (
                  <div
                    key={lang.code}
                    className={`flex items-center justify-between p-2 rounded cursor-pointer ${
                      selectedLanguage === lang.code ? "bg-blue-50 border border-blue-200" : "hover:bg-gray-50"
                    }`}
                    onClick={() => setSelectedLanguage(lang.code)}
                  >
                    <span className="text-sm">
                      {lang.flag} {lang.name}
                    </span>
                    {selectedLanguage === lang.code && (
                      <Badge variant="default" className="text-xs">
                        Active
                      </Badge>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Quick Test Messages</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {[
                "Hello, how are you today?",
                "What should I do during a flood?",
                "Tell me about earthquake safety.",
                "How can I prepare for extreme weather?",
              ].map((testMessage, index) => (
                <Button
                  key={index}
                  variant="outline"
                  className="w-full justify-start text-sm bg-transparent"
                  onClick={() => handleTextToSpeech(testMessage, `test-${index}`)}
                  disabled={isSpeaking}
                >
                  <Volume2 className="h-3 w-3 mr-2" />
                  {testMessage}
                </Button>
              ))}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Audio Debugging</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button
                variant="outline"
                className="w-full justify-start text-sm bg-transparent"
                onClick={() => {
                  // Test Windows Speech Synthesis API
                  if ("speechSynthesis" in window) {
                    const utterance = new SpeechSynthesisUtterance("Testing Windows speech synthesis")
                    window.speechSynthesis.speak(utterance)
                  } else {
                    setTtsError("Speech Synthesis not supported on this browser")
                  }
                }}
              >
                🔊 Test Windows Speech API
              </Button>

              <Button
                variant="outline"
                className="w-full justify-start text-sm bg-transparent"
                onClick={() => {
                  // Test audio context and supported formats
                  try {
                    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
                    console.log("Audio Context State:", audioContext.state)

                    // Test supported audio formats
                    const audio = new Audio()
                    const formats = {
                      WAV: audio.canPlayType("audio/wav"),
                      MP3: audio.canPlayType("audio/mpeg"),
                      OGG: audio.canPlayType("audio/ogg"),
                      AAC: audio.canPlayType("audio/aac"),
                      WEBM: audio.canPlayType("audio/webm"),
                    }

                    console.log("Supported audio formats:", formats)
                    setTtsError(`Audio Context: ${audioContext.state}. Supported formats: ${JSON.stringify(formats)}`)
                  } catch (e) {
                    setTtsError("Audio Context not supported")
                  }
                }}
              >
                🎵 Test Audio Formats
              </Button>

              <Button
                variant="outline"
                className="w-full justify-start text-sm bg-transparent"
                onClick={async () => {
                  // Test direct audio file fetch
                  if (backendConnected) {
                    try {
                      const response = await fetch(`${backendUrl}/api/text-to-speech`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ text: "Test", language: "en", speed: 150 }),
                      })

                      if (response.ok) {
                        const data = await response.json()
                        console.log("Test TTS response:", data)

                        if (data.audio_url) {
                          const audioResponse = await fetch(`${backendUrl}${data.audio_url}`)
                          const blob = await audioResponse.blob()
                          console.log("Test audio blob:", blob.size, blob.type)
                          setTtsError(`Test successful. Audio: ${blob.size} bytes, type: ${blob.type}`)
                        }
                      } else {
                        setTtsError(`Test failed: ${response.status}`)
                      }
                    } catch (e) {
                      if (e instanceof Error) {
                        setTtsError(`Test error: ${e.message}`);
                      } else {
                        setTtsError("Test error: Unknown error");
                      }
                    }
                    
                  } else {
                    setTtsError("Backend not connected")
                  }
                }}
              >
                🧪 Test TTS Endpoint
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
