"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import {
  AlertTriangle,
  Users,
  Shield,
  MessageSquare,
  Activity,
  Thermometer,
  Wind,
  Droplets,
  Sun,
  Zap,
  Database,
} from "lucide-react"
import WeatherWidget from "./components/weather-widget"
import CitizenReporting from "./components/citizen-reporting"
import VoiceChat from "./components/voice-chat"
import RegionalAnalysis from "./components/regional-analysis"
import EmergencyContacts from "./components/emergency-contacts"
import BlockchainDashboard from "./components/blockchain-dashboard"
import BlockchainAnalytics from "./components/blockchain-analytics"
import QuantumOptimizer from "./components/quantum-optimizer"
import PolicyEngine from "./components/policy-engine"
import ResilienceAnalyzer from "./components/resilience-analyzer"

export default function Dashboard() {
  const [systemStatus, setSystemStatus] = useState({
    ai_agents: "online",
    quantum_optimizer: "ready",
    blockchain: "active",
    rag_system: "loaded",
    weather_service: "connected",
    backend_connected: false,
  })

  const [climateMetrics, setClimateMetrics] = useState({
    temperature: 32,
    aqi: 156,
    rainfall_deficit: 25,
    renewable_percent: 68.5,
  })

  const [resilienceScore, setResilienceScore] = useState(72)

  // Check if we're running in development mode
  const isDevelopment = process.env.NODE_ENV === "development"
  const backendUrl = isDevelopment ? "http://localhost:8000" : ""

  // Fetch system health on component mount
  useEffect(() => {
    const fetchSystemHealth = async () => {
      try {
        // Try to connect to the backend health endpoint
        const response = await fetch(`${backendUrl}/health`, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        })

        if (response.ok) {
          const data = await response.json()
          console.log("Backend health check passed:", data)
          setSystemStatus((prev) => ({ ...prev, backend_connected: true }))
        } else {
          console.warn("Backend health check failed with status:", response.status)
          setSystemStatus((prev) => ({ ...prev, backend_connected: false }))
        }
      } catch (error) {
        console.warn("Backend not available, running in demo mode:", error.message)
        setSystemStatus((prev) => ({ ...prev, backend_connected: false }))
      }
    }

    fetchSystemHealth()

    // Set up periodic health checks every 30 seconds
    const healthCheckInterval = setInterval(fetchSystemHealth, 30000)

    return () => clearInterval(healthCheckInterval)
  }, [backendUrl])

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-green-50 to-purple-50">
      {/* Enhanced Header */}
      <header className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <div className="bg-gradient-to-r from-blue-600 via-green-600 to-purple-600 p-3 rounded-xl">
                <Shield className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent">
                  ClimaX
                </h1>
                <p className="text-sm text-gray-600">AI + Quantum + Blockchain Climate Resilience </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div
                    className={`w-2 h-2 rounded-full animate-pulse ${systemStatus.backend_connected ? "bg-green-500" : "bg-yellow-500"}`}
                  ></div>
                  <div
                    className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"
                    style={{ animationDelay: "0.4s" }}
                  ></div>
                </div>
                <Badge
                  variant="outline"
                  className={`${systemStatus.backend_connected ? "bg-green-50 text-green-700 border-green-200" : "bg-yellow-50 text-yellow-700 border-yellow-200"}`}
                >
                  {systemStatus.backend_connected ? "Backend Connected" : "Demo Mode"}
                </Badge>
              </div>
              <Button variant="outline" size="sm">
                <MessageSquare className="h-4 w-4 mr-2" />
                Support
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* System Status Banner */}
        <Alert
          className={`mb-6 ${systemStatus.backend_connected ? "border-blue-200 bg-gradient-to-r from-blue-50 to-green-50" : "border-yellow-200 bg-gradient-to-r from-yellow-50 to-orange-50"}`}
        >
          <Activity className={`h-4 w-4 ${systemStatus.backend_connected ? "text-blue-600" : "text-yellow-600"}`} />
          <AlertTitle className={systemStatus.backend_connected ? "text-blue-800" : "text-yellow-800"}>
            System Status: {systemStatus.backend_connected ? "All Modules Active" : "Running in Demo Mode"}
          </AlertTitle>
          <AlertDescription className={systemStatus.backend_connected ? "text-blue-700" : "text-yellow-700"}>
            {systemStatus.backend_connected ? (
              <>
                AI Agents, Quantum Optimizer, Blockchain, RAG System, and Weather Service are all operational.
                <div className="flex space-x-4 mt-2 text-xs">
                  <span className="flex items-center">
                    <Zap className="h-3 w-3 mr-1" />
                    Quantum Ready
                  </span>
                  <span className="flex items-center">
                    <Database className="h-3 w-3 mr-1" />
                    Blockchain Active
                  </span>
                  <span className="flex items-center">
                    <MessageSquare className="h-3 w-3 mr-1" />
                    AI Agents Online
                  </span>
                </div>
              </>
            ) : (
              <>
                Backend services are not available. The application is running with demo data and limited functionality.
                <div className="flex space-x-4 mt-2 text-xs">
                  <span className="flex items-center">
                    <Zap className="h-3 w-3 mr-1" />
                    Demo Data
                  </span>
                  <span className="flex items-center">
                    <Database className="h-3 w-3 mr-1" />
                    Mock Services
                  </span>
                  <span className="flex items-center">
                    <MessageSquare className="h-3 w-3 mr-1" />
                    Limited Features
                  </span>
                </div>
              </>
            )}
          </AlertDescription>
        </Alert>

        {/* Enhanced Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="bg-gradient-to-r from-blue-500 to-blue-600 text-white transform hover:scale-105 transition-transform">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium opacity-90">Temperature</CardTitle>
              <Thermometer className="h-4 w-4 opacity-90" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{climateMetrics.temperature}°C</div>
              <p className="text-xs opacity-90">+2°C from yesterday</p>
              <div className="mt-2 h-1 bg-white/20 rounded-full">
                <div className="h-1 bg-white rounded-full" style={{ width: "65%" }}></div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-r from-red-500 to-red-600 text-white transform hover:scale-105 transition-transform">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium opacity-90">Air Quality Index</CardTitle>
              <Wind className="h-4 w-4 opacity-90" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{climateMetrics.aqi}</div>
              <p className="text-xs opacity-90">Moderate - Unhealthy</p>
              <div className="mt-2 h-1 bg-white/20 rounded-full">
                <div className="h-1 bg-white rounded-full" style={{ width: "78%" }}></div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-r from-orange-500 to-orange-600 text-white transform hover:scale-105 transition-transform">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium opacity-90">Rainfall Deficit</CardTitle>
              <Droplets className="h-4 w-4 opacity-90" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{climateMetrics.rainfall_deficit}%</div>
              <p className="text-xs opacity-90">Below normal levels</p>
              <div className="mt-2 h-1 bg-white/20 rounded-full">
                <div className="h-1 bg-white rounded-full" style={{ width: "25%" }}></div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-r from-green-500 to-green-600 text-white transform hover:scale-105 transition-transform">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium opacity-90">Renewable Energy</CardTitle>
              <Sun className="h-4 w-4 opacity-90" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{climateMetrics.renewable_percent}%</div>
              <p className="text-xs opacity-90">Of total energy mix</p>
              <div className="mt-2 h-1 bg-white/20 rounded-full">
                <div className="h-1 bg-white rounded-full" style={{ width: "68%" }}></div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Enhanced Resilience Score */}
        <Card className="mb-8 bg-gradient-to-r from-white to-gray-50">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5 text-blue-600" />
              <span>AI-Powered Climate Resilience Score</span>
              <Badge variant="outline" className="ml-2">
                {systemStatus.backend_connected ? "Gemini AI" : "Demo"}
              </Badge>
            </CardTitle>
            <CardDescription>
              Comprehensive assessment using AI agents, quantum optimization, and blockchain verification
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-4 mb-6">
              <div className="flex-1">
                <Progress value={resilienceScore} className="h-4" />
              </div>
              <div className="text-3xl font-bold text-blue-600">{resilienceScore}/100</div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {[
                { label: "Infrastructure", score: 78, color: "bg-blue-500" },
                { label: "Economic", score: 65, color: "bg-green-500" },
                { label: "Social", score: 82, color: "bg-purple-500" },
                { label: "Environmental", score: 58, color: "bg-orange-500" },
                { label: "Governance", score: 75, color: "bg-red-500" },
                { label: "Emergency", score: 70, color: "bg-yellow-500" },
              ].map((category) => (
                <div key={category.label} className="text-center p-3 bg-white rounded-lg shadow-sm">
                  <div className="text-sm font-medium text-gray-600 mb-2">{category.label}</div>
                  <div className="text-xl font-bold text-gray-900 mb-2">{category.score}</div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className={`h-2 rounded-full ${category.color}`} style={{ width: `${category.score}%` }}></div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Enhanced Main Content Tabs */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3 lg:grid-cols-9 bg-white shadow-sm">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="weather">Weather</TabsTrigger>
            <TabsTrigger value="reporting">Reporting</TabsTrigger>
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
            <TabsTrigger value="voice">Voice AI</TabsTrigger>
            <TabsTrigger value="emergency">Emergency</TabsTrigger>
            <TabsTrigger value="blockchain">Blockchain</TabsTrigger>
            <TabsTrigger value="quantum">Quantum</TabsTrigger>
            <TabsTrigger value="policy">Policy</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-gradient-to-br from-orange-50 to-red-50">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <AlertTriangle className="h-5 w-5 text-orange-600" />
                    <span>AI-Generated Alerts</span>
                    <Badge variant="outline">{systemStatus.backend_connected ? "Real-time" : "Demo"}</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {[
                    {
                      type: "Heat Wave",
                      region: "Northern Districts",
                      severity: "High",
                      time: "2 hours ago",
                      ai: systemStatus.backend_connected ? "Gemini AI" : "Demo Data",
                    },
                    {
                      type: "Air Quality",
                      region: "Urban Areas",
                      severity: "Moderate",
                      time: "4 hours ago",
                      ai: systemStatus.backend_connected ? "RAG System" : "Demo Data",
                    },
                    {
                      type: "Drought",
                      region: "Agricultural Zones",
                      severity: "Medium",
                      time: "1 day ago",
                      ai: systemStatus.backend_connected ? "Climate Agent" : "Demo Data",
                    },
                  ].map((alert, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-white rounded-lg shadow-sm">
                      <div>
                        <div className="font-medium">
                          {alert.type} - {alert.region}
                        </div>
                        <div className="text-sm text-gray-600 flex items-center space-x-2">
                          <span>{alert.time}</span>
                          <Badge variant="outline" className="text-xs">
                            {alert.ai}
                          </Badge>
                        </div>
                      </div>
                      <Badge
                        variant={
                          alert.severity === "High"
                            ? "destructive"
                            : alert.severity === "Moderate"
                              ? "default"
                              : "secondary"
                        }
                      >
                        {alert.severity}
                      </Badge>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-blue-50 to-green-50">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Users className="h-5 w-5 text-blue-600" />
                    <span>Blockchain-Verified Reports</span>
                    <Badge variant="outline">{systemStatus.backend_connected ? "Immutable" : "Demo"}</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {[
                    {
                      type: "Flood",
                      location: "Downtown Area",
                      status: "Verified",
                      time: "30 min ago",
                      hash: "0x1a2b3c",
                    },
                    {
                      type: "Heat Stress",
                      location: "Industrial Zone",
                      status: "Pending",
                      time: "1 hour ago",
                      hash: "0x4d5e6f",
                    },
                    {
                      type: "Air Pollution",
                      location: "Residential Area",
                      status: "Verified",
                      time: "2 hours ago",
                      hash: "0x7g8h9i",
                    },
                  ].map((report, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-white rounded-lg shadow-sm">
                      <div>
                        <div className="font-medium">
                          {report.type} - {report.location}
                        </div>
                        <div className="text-sm text-gray-600 flex items-center space-x-2">
                          <span>{report.time}</span>
                          <code className="text-xs bg-gray-100 px-1 rounded">{report.hash}</code>
                        </div>
                      </div>
                      <Badge variant={report.status === "Verified" ? "default" : "secondary"}>{report.status}</Badge>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>

            {/* System Modules Status */}
            <Card>
              <CardHeader>
                <CardTitle>System Modules Status</CardTitle>
                <CardDescription>Real-time status of all ClimaX components</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                  {[
                    {
                      name: "AI Agents",
                      status: systemStatus.backend_connected ? "Online" : "Demo",
                      icon: MessageSquare,
                      color: systemStatus.backend_connected ? "text-green-600" : "text-yellow-600",
                    },
                    {
                      name: "Quantum Optimizer",
                      status: systemStatus.backend_connected ? "Ready" : "Demo",
                      icon: Zap,
                      color: systemStatus.backend_connected ? "text-blue-600" : "text-yellow-600",
                    },
                    {
                      name: "Blockchain",
                      status: systemStatus.backend_connected ? "Active" : "Demo",
                      icon: Database,
                      color: systemStatus.backend_connected ? "text-purple-600" : "text-yellow-600",
                    },
                    {
                      name: "RAG System",
                      status: systemStatus.backend_connected ? "Loaded" : "Demo",
                      icon: Activity,
                      color: systemStatus.backend_connected ? "text-orange-600" : "text-yellow-600",
                    },
                    {
                      name: "Weather Service",
                      status: systemStatus.backend_connected ? "Connected" : "Demo",
                      icon: Sun,
                      color: systemStatus.backend_connected ? "text-yellow-600" : "text-gray-600",
                    },
                  ].map((module, index) => (
                    <div key={index} className="text-center p-4 bg-gray-50 rounded-lg">
                      <module.icon className={`h-8 w-8 mx-auto mb-2 ${module.color}`} />
                      <div className="font-medium">{module.name}</div>
                      <Badge variant="outline" className="mt-1">
                        {module.status}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="weather">
            <WeatherWidget backendConnected={systemStatus.backend_connected} backendUrl={backendUrl} />
          </TabsContent>

          <TabsContent value="reporting">
            <CitizenReporting backendConnected={systemStatus.backend_connected} backendUrl={backendUrl} />
          </TabsContent>

          <TabsContent value="analysis">
            <Tabs defaultValue="regional" className="space-y-6">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="regional">Regional Analysis</TabsTrigger>
                <TabsTrigger value="resilience">Resilience Analyzer</TabsTrigger>
              </TabsList>
              <TabsContent value="regional">
                <RegionalAnalysis backendConnected={systemStatus.backend_connected} backendUrl={backendUrl} />
              </TabsContent>
              <TabsContent value="resilience">
                <ResilienceAnalyzer backendConnected={systemStatus.backend_connected} backendUrl={backendUrl} />
              </TabsContent>
            </Tabs>
          </TabsContent>

          <TabsContent value="voice">
            <VoiceChat backendConnected={systemStatus.backend_connected} backendUrl={backendUrl} />
          </TabsContent>

          <TabsContent value="emergency">
            <EmergencyContacts backendConnected={systemStatus.backend_connected} backendUrl={backendUrl} />
          </TabsContent>

          <TabsContent value="blockchain" className="space-y-6">
            <Tabs defaultValue="dashboard" className="space-y-6">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="dashboard">Blockchain Dashboard</TabsTrigger>
                <TabsTrigger value="analytics">Analytics</TabsTrigger>
              </TabsList>

              <TabsContent value="dashboard">
                <BlockchainDashboard backendConnected={systemStatus.backend_connected} backendUrl={backendUrl} />
              </TabsContent>

              <TabsContent value="analytics">
                <BlockchainAnalytics backendConnected={systemStatus.backend_connected} backendUrl={backendUrl} />
              </TabsContent>
            </Tabs>
          </TabsContent>

          <TabsContent value="quantum">
            <QuantumOptimizer backendConnected={systemStatus.backend_connected} backendUrl={backendUrl} />
          </TabsContent>

          <TabsContent value="policy">
            <PolicyEngine backendConnected={systemStatus.backend_connected} backendUrl={backendUrl} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
