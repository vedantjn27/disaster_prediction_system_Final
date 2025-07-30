"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { MapPin, BarChart3, TrendingUp, Thermometer, Wind, Droplets, Zap } from "lucide-react"

interface RegionalAnalysisProps {
  backendConnected: boolean
  backendUrl: string
}

export default function RegionalAnalysis({ backendConnected, backendUrl }: RegionalAnalysisProps) {
  const [selectedRegion, setSelectedRegion] = useState("Delhi")
  const [isLoading, setIsLoading] = useState(false)
  const [analysisData, setAnalysisData] = useState({
    Delhi: {
      climate_type: "Semi-arid",
      major_issues: ["Air pollution", "Heat waves", "Water scarcity", "Urban heat island"],
      population: "32 million",
      key_sectors: ["Transportation", "Industry", "Power generation", "Construction"],
      current_aqi: 287,
      avg_temp: 42,
      rainfall_deficit: 45,
      renewable_percent: 8.5,
    },
    Mumbai: {
      climate_type: "Tropical",
      major_issues: ["Flooding", "Sea level rise", "Coastal erosion", "Air pollution"],
      population: "20 million",
      key_sectors: ["Finance", "Entertainment", "Textiles", "Chemicals"],
      current_aqi: 156,
      avg_temp: 34,
      rainfall_deficit: 15,
      renewable_percent: 12.3,
    },
    Bangalore: {
      climate_type: "Tropical savanna",
      major_issues: ["Water scarcity", "Urban heat island", "Traffic pollution", "Waste management"],
      population: "12 million",
      key_sectors: ["IT", "Biotechnology", "Aerospace", "Electronics"],
      current_aqi: 98,
      avg_temp: 28,
      rainfall_deficit: 25,
      renewable_percent: 18.7,
    },
  })

  const regions = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"]

  const handleRegionChange = async (region: string) => {
    setSelectedRegion(region)
    setIsLoading(true)

    try {
      if (backendConnected) {
        // Call the real backend API
        const response = await fetch(`${backendUrl}/region/${region}/data`)
        if (response.ok) {
          const data = await response.json()
          setAnalysisData((prev) => ({ ...prev, [region]: data }))
        }
      } else {
        // Simulate API call
        await new Promise((resolve) => setTimeout(resolve, 1000))
      }
    } catch (error) {
      console.error("Failed to fetch region data:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const currentData = analysisData[selectedRegion] || analysisData.Delhi

  const getAQIColor = (aqi: number) => {
    if (aqi <= 50) return "text-green-600"
    if (aqi <= 100) return "text-yellow-600"
    if (aqi <= 150) return "text-orange-600"
    return "text-red-600"
  }

  const getAQIBadge = (aqi: number) => {
    if (aqi <= 50) return { variant: "default" as const, text: "Good" }
    if (aqi <= 100) return { variant: "secondary" as const, text: "Moderate" }
    if (aqi <= 150) return { variant: "destructive" as const, text: "Unhealthy" }
    return { variant: "destructive" as const, text: "Hazardous" }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="bg-gradient-to-r from-blue-50 to-green-50">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <MapPin className="h-6 w-6 text-blue-600" />
            <span>Regional Climate Analysis</span>
            <Badge variant="outline" className="bg-blue-100 text-blue-700">
              {backendConnected ? "AI-Powered" : "Demo Mode"}
            </Badge>
          </CardTitle>
          <CardDescription>
            {backendConnected
              ? "Comprehensive climate analysis powered by AI agents and real-time data"
              : "Regional analysis running in demo mode (Backend not connected)"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4">
            <Select value={selectedRegion} onValueChange={handleRegionChange}>
              <SelectTrigger className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {regions.map((region) => (
                  <SelectItem key={region} value={region}>
                    {region}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {isLoading && (
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                <span>Loading analysis...</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {!backendConnected && (
        <Alert className="border-yellow-200 bg-yellow-50">
          <MapPin className="h-4 w-4 text-yellow-600" />
          <AlertDescription className="text-yellow-700">
            Backend not connected. Showing demo data for regional analysis.
          </AlertDescription>
        </Alert>
      )}

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-gradient-to-r from-red-500 to-red-600 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium opacity-90">Air Quality Index</CardTitle>
            <Wind className="h-4 w-4 opacity-90" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{currentData.current_aqi}</div>
            <div className="flex items-center space-x-2 mt-2">
              <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
                {getAQIBadge(currentData.current_aqi).text}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-r from-orange-500 to-orange-600 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium opacity-90">Temperature</CardTitle>
            <Thermometer className="h-4 w-4 opacity-90" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{currentData.avg_temp}°C</div>
            <p className="text-xs opacity-90">Average high temperature</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-r from-blue-500 to-blue-600 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium opacity-90">Rainfall Deficit</CardTitle>
            <Droplets className="h-4 w-4 opacity-90" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{currentData.rainfall_deficit}%</div>
            <p className="text-xs opacity-90">Below normal levels</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-r from-green-500 to-green-600 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium opacity-90">Renewable Energy</CardTitle>
            <Zap className="h-4 w-4 opacity-90" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{currentData.renewable_percent}%</div>
            <p className="text-xs opacity-90">Of total energy mix</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="climate">Climate Data</TabsTrigger>
          <TabsTrigger value="risks">Risk Assessment</TabsTrigger>
          <TabsTrigger value="strategies">Mitigation</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <MapPin className="h-5 w-5 text-blue-600" />
                  <span>Region Profile</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-sm font-medium text-gray-600">Climate Type:</span>
                    <div className="font-medium">{currentData.climate_type}</div>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-600">Population:</span>
                    <div className="font-medium">{currentData.population}</div>
                  </div>
                </div>

                <div>
                  <span className="text-sm font-medium text-gray-600 block mb-2">Key Economic Sectors:</span>
                  <div className="flex flex-wrap gap-2">
                    {currentData.key_sectors.map((sector, index) => (
                      <Badge key={index} variant="outline">
                        {sector}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="h-5 w-5 text-orange-600" />
                  <span>Major Climate Issues</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {currentData.major_issues.map((issue, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <span className="font-medium">{issue}</span>
                      <Badge variant="destructive">High Priority</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="climate" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Climate Indicators</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium">Air Quality Index</span>
                    <span className={`text-sm font-bold ${getAQIColor(currentData.current_aqi)}`}>
                      {currentData.current_aqi}
                    </span>
                  </div>
                  <Progress value={Math.min((currentData.current_aqi / 500) * 100, 100)} className="h-2" />
                </div>

                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium">Temperature Stress</span>
                    <span className="text-sm font-bold text-orange-600">{currentData.avg_temp}°C</span>
                  </div>
                  <Progress value={(currentData.avg_temp / 50) * 100} className="h-2" />
                </div>

                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium">Rainfall Deficit</span>
                    <span className="text-sm font-bold text-blue-600">{currentData.rainfall_deficit}%</span>
                  </div>
                  <Progress value={currentData.rainfall_deficit} className="h-2" />
                </div>

                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium">Renewable Energy</span>
                    <span className="text-sm font-bold text-green-600">{currentData.renewable_percent}%</span>
                  </div>
                  <Progress value={currentData.renewable_percent} className="h-2" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Historical Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 bg-red-50 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <TrendingUp className="h-4 w-4 text-red-600" />
                      <span className="font-medium text-red-800">Temperature Rising</span>
                    </div>
                    <p className="text-sm text-red-700">+2.3°C increase over the last decade</p>
                  </div>

                  <div className="p-4 bg-orange-50 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <Wind className="h-4 w-4 text-orange-600" />
                      <span className="font-medium text-orange-800">Air Quality Declining</span>
                    </div>
                    <p className="text-sm text-orange-700">AQI increased by 45% in 5 years</p>
                  </div>

                  <div className="p-4 bg-blue-50 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <Droplets className="h-4 w-4 text-blue-600" />
                      <span className="font-medium text-blue-800">Rainfall Patterns Changing</span>
                    </div>
                    <p className="text-sm text-blue-700">Irregular monsoon patterns observed</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="risks" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <BarChart3 className="h-5 w-5 text-red-600" />
                <span>Risk Assessment Matrix</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium mb-4">High Risk Factors</h4>
                  <div className="space-y-3">
                    {[
                      { risk: "Air Pollution", probability: 95, impact: 85 },
                      { risk: "Heat Waves", probability: 80, impact: 75 },
                      { risk: "Water Scarcity", probability: 70, impact: 80 },
                      { risk: "Urban Heat Island", probability: 85, impact: 65 },
                    ].map((item, index) => (
                      <div key={index} className="p-3 border rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-medium">{item.risk}</span>
                          <Badge variant="destructive">High</Badge>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-gray-600">Probability: </span>
                            <span className="font-medium">{item.probability}%</span>
                          </div>
                          <div>
                            <span className="text-gray-600">Impact: </span>
                            <span className="font-medium">{item.impact}%</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-4">Vulnerability Assessment</h4>
                  <div className="space-y-4">
                    <div>
                      <span className="text-sm font-medium">Population Vulnerability</span>
                      <Progress value={75} className="mt-1" />
                      <span className="text-xs text-gray-600">75% - High density urban areas</span>
                    </div>
                    <div>
                      <span className="text-sm font-medium">Infrastructure Resilience</span>
                      <Progress value={45} className="mt-1" />
                      <span className="text-xs text-gray-600">45% - Aging infrastructure</span>
                    </div>
                    <div>
                      <span className="text-sm font-medium">Economic Impact</span>
                      <Progress value={80} className="mt-1" />
                      <span className="text-xs text-gray-600">80% - High economic exposure</span>
                    </div>
                    <div>
                      <span className="text-sm font-medium">Environmental Degradation</span>
                      <Progress value={85} className="mt-1" />
                      <span className="text-xs text-gray-600">85% - Severe environmental stress</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="strategies" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Zap className="h-5 w-5 text-green-600" />
                  <span>Mitigation Strategies</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    {
                      strategy: "Expand Metro Network",
                      impact: "Reduce vehicular emissions by 30%",
                      timeline: "5 years",
                      cost: "High",
                    },
                    {
                      strategy: "Rooftop Solar Program",
                      impact: "Increase renewable energy to 25%",
                      timeline: "3 years",
                      cost: "Medium",
                    },
                    {
                      strategy: "Urban Forest Initiative",
                      impact: "Reduce temperature by 2°C",
                      timeline: "2 years",
                      cost: "Low",
                    },
                    {
                      strategy: "Waste-to-Energy Plants",
                      impact: "Reduce landfill waste by 60%",
                      timeline: "4 years",
                      cost: "High",
                    },
                  ].map((item, index) => (
                    <div key={index} className="p-4 border rounded-lg">
                      <div className="flex justify-between items-start mb-2">
                        <h5 className="font-medium">{item.strategy}</h5>
                        <Badge
                          variant={
                            item.cost === "High" ? "destructive" : item.cost === "Medium" ? "secondary" : "default"
                          }
                        >
                          {item.cost} Cost
                        </Badge>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{item.impact}</p>
                      <div className="text-xs text-gray-500">Timeline: {item.timeline}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="h-5 w-5 text-blue-600" />
                  <span>Implementation Roadmap</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 bg-green-50 rounded-lg">
                    <h5 className="font-medium text-green-800 mb-2">Phase 1 (0-2 years)</h5>
                    <ul className="text-sm text-green-700 space-y-1">
                      <li>• Urban forest plantation drive</li>
                      <li>• Air quality monitoring network</li>
                      <li>• Public awareness campaigns</li>
                    </ul>
                  </div>

                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h5 className="font-medium text-blue-800 mb-2">Phase 2 (2-5 years)</h5>
                    <ul className="text-sm text-blue-700 space-y-1">
                      <li>• Rooftop solar installations</li>
                      <li>• Electric bus fleet expansion</li>
                      <li>• Waste management upgrades</li>
                    </ul>
                  </div>

                  <div className="p-4 bg-purple-50 rounded-lg">
                    <h5 className="font-medium text-purple-800 mb-2">Phase 3 (5+ years)</h5>
                    <ul className="text-sm text-purple-700 space-y-1">
                      <li>• Metro network expansion</li>
                      <li>• Smart city infrastructure</li>
                      <li>• Carbon neutrality targets</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
