"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { FileText, TrendingUp, Users, Building, Lightbulb, BarChart3, CheckCircle } from "lucide-react"

export default function PolicyEngine() {
  const [isGenerating, setIsGenerating] = useState(false)
  const [policyResults, setPolicyResults] = useState(null)
  const [disasterPatterns, setDisasterPatterns] = useState([
    { region: "Kerala", disaster_type: "Flood", frequency: 7 },
    { region: "Gujarat", disaster_type: "Earthquake", frequency: 3 },
    { region: "Assam", disaster_type: "Flood", frequency: 8 },
    { region: "Punjab", disaster_type: "Drought", frequency: 2 },
  ])

  const [newPattern, setNewPattern] = useState({
    region: "",
    disaster_type: "",
    frequency: "",
  })

  const handleGeneratePolicies = async () => {
    setIsGenerating(true)

    try {
      // Simulate API call to policy engine
      await new Promise((resolve) => setTimeout(resolve, 3000))

      // Mock policy generation results
      const mockResults = disasterPatterns.map((pattern) => ({
        region: pattern.region,
        disaster_type: pattern.disaster_type,
        frequency: pattern.frequency,
        current_policies: getCurrentPolicies(pattern.disaster_type),
        evidence_based_recommendation: generateRecommendation(pattern),
        factors_considered: getRegionalFactors(pattern.region),
        implementation_timeline: "6-12 months",
        budget_estimate: `₹${Math.floor(Math.random() * 500 + 100)} crores`,
        expected_impact: `${Math.floor(Math.random() * 30 + 40)}% reduction in disaster impact`,
      }))

      setPolicyResults(mockResults)
    } catch (error) {
      console.error("Policy generation failed:", error)
    } finally {
      setIsGenerating(false)
    }
  }

  const getCurrentPolicies = (disasterType) => {
    const policies = {
      Flood: ["Evacuation drills every 6 months", "Flood forecasting system via SMS alerts"],
      Earthquake: ["Building codes for seismic zones", "School-level earthquake drills"],
      Cyclone: ["Cyclone shelters in coastal areas", "Disaster communication vans for early warning"],
      Drought: ["Water conservation mandates", "Crop insurance schemes"],
    }
    return policies[disasterType] || ["No standard policies found"]
  }

  const generateRecommendation = (pattern) => {
    if (pattern.frequency >= 5) {
      return `Due to frequent ${pattern.disaster_type} events in ${pattern.region}, implement AI-based early warning systems, install real-time monitoring sensors, and develop mobile-first alert apps for citizens.`
    } else {
      return `In ${pattern.region}, occasional ${pattern.disaster_type} occurrences should be addressed with community awareness programs, emergency preparedness training, and pre-positioned emergency kits at local centers.`
    }
  }

  const getRegionalFactors = (region) => {
    const factors = {
      Kerala: ["High rainfall zone", "Dense population", "Riverine landscape"],
      Gujarat: ["Seismic zone", "Dry terrain", "Coastal region"],
      Assam: ["Hilly terrain", "Heavy monsoon", "Poor infrastructure"],
      Punjab: ["Plains", "Low disaster frequency", "Good road connectivity"],
    }
    return factors[region] || ["General rural area"]
  }

  const addDisasterPattern = () => {
    if (newPattern.region && newPattern.disaster_type && newPattern.frequency) {
      setDisasterPatterns([
        ...disasterPatterns,
        {
          region: newPattern.region,
          disaster_type: newPattern.disaster_type,
          frequency: Number.parseInt(newPattern.frequency),
        },
      ])
      setNewPattern({ region: "", disaster_type: "", frequency: "" })
    }
  }

  const removePattern = (index) => {
    setDisasterPatterns(disasterPatterns.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-6">
      {/* Policy Engine Header */}
      <Card className="bg-gradient-to-r from-green-50 to-blue-50">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <FileText className="h-6 w-6 text-green-600" />
            <span>Evidence-Based Policy Engine</span>
            <Badge variant="outline" className="bg-green-100 text-green-700">
              AI-Powered
            </Badge>
          </CardTitle>
          <CardDescription>
            Generate data-driven policy recommendations based on disaster patterns and regional factors
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-white rounded-lg">
              <TrendingUp className="h-6 w-6 text-blue-500 mx-auto mb-2" />
              <div className="text-lg font-bold">{disasterPatterns.length}</div>
              <div className="text-sm text-gray-600">Active Patterns</div>
            </div>
            <div className="text-center p-3 bg-white rounded-lg">
              <Users className="h-6 w-6 text-green-500 mx-auto mb-2" />
              <div className="text-lg font-bold">{new Set(disasterPatterns.map((p) => p.region)).size}</div>
              <div className="text-sm text-gray-600">Regions Covered</div>
            </div>
            <div className="text-center p-3 bg-white rounded-lg">
              <Building className="h-6 w-6 text-orange-500 mx-auto mb-2" />
              <div className="text-lg font-bold">{new Set(disasterPatterns.map((p) => p.disaster_type)).size}</div>
              <div className="text-sm text-gray-600">Disaster Types</div>
            </div>
            <div className="text-center p-3 bg-white rounded-lg">
              <Lightbulb className="h-6 w-6 text-purple-500 mx-auto mb-2" />
              <div className="text-lg font-bold">{policyResults?.length || 0}</div>
              <div className="text-sm text-gray-600">Policies Generated</div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="patterns" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="patterns">Disaster Patterns</TabsTrigger>
          <TabsTrigger value="generator">Policy Generator</TabsTrigger>
          <TabsTrigger value="results">Policy Results</TabsTrigger>
        </TabsList>

        <TabsContent value="patterns" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Add New Pattern */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="h-5 w-5 text-blue-600" />
                  <span>Add Disaster Pattern</span>
                </CardTitle>
                <CardDescription>Input historical disaster data for policy analysis</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Region</label>
                  <Input
                    placeholder="Enter region name"
                    value={newPattern.region}
                    onChange={(e) => setNewPattern((prev) => ({ ...prev, region: e.target.value }))}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Disaster Type</label>
                  <Select
                    value={newPattern.disaster_type}
                    onValueChange={(value) => setNewPattern((prev) => ({ ...prev, disaster_type: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select disaster type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Flood">Flood</SelectItem>
                      <SelectItem value="Earthquake">Earthquake</SelectItem>
                      <SelectItem value="Cyclone">Cyclone</SelectItem>
                      <SelectItem value="Drought">Drought</SelectItem>
                      <SelectItem value="Heatwave">Heatwave</SelectItem>
                      <SelectItem value="Landslide">Landslide</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Frequency (per decade)</label>
                  <Input
                    type="number"
                    placeholder="Enter frequency"
                    value={newPattern.frequency}
                    onChange={(e) => setNewPattern((prev) => ({ ...prev, frequency: e.target.value }))}
                  />
                </div>
                <Button onClick={addDisasterPattern} className="w-full">
                  Add Pattern
                </Button>
              </CardContent>
            </Card>

            {/* Current Patterns */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="h-5 w-5 text-green-600" />
                  <span>Current Patterns</span>
                </CardTitle>
                <CardDescription>Historical disaster patterns for analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {disasterPatterns.map((pattern, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <div className="font-medium">{pattern.region}</div>
                        <div className="text-sm text-gray-600">
                          {pattern.disaster_type} • {pattern.frequency} times/decade
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant={pattern.frequency >= 5 ? "destructive" : "secondary"}>
                          {pattern.frequency >= 5 ? "High Risk" : "Moderate Risk"}
                        </Badge>
                        <Button variant="outline" size="sm" onClick={() => removePattern(index)}>
                          Remove
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="generator" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Lightbulb className="h-5 w-5 text-purple-600" />
                <span>AI Policy Generator</span>
              </CardTitle>
              <CardDescription>
                Generate evidence-based policies using machine learning and regional analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="text-center space-y-4">
                <Button
                  onClick={handleGeneratePolicies}
                  disabled={isGenerating || disasterPatterns.length === 0}
                  size="lg"
                  className="w-full bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700"
                >
                  {isGenerating ? (
                    <>
                      <TrendingUp className="h-4 w-4 mr-2 animate-spin" />
                      Generating Policies...
                    </>
                  ) : (
                    <>
                      <Lightbulb className="h-4 w-4 mr-2" />
                      Generate Evidence-Based Policies
                    </>
                  )}
                </Button>

                {isGenerating && (
                  <Alert className="border-blue-200 bg-blue-50">
                    <TrendingUp className="h-4 w-4 text-blue-600" />
                    <AlertDescription className="text-blue-700">
                      Analyzing disaster patterns, regional factors, and current policies to generate evidence-based
                      recommendations...
                    </AlertDescription>
                  </Alert>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium mb-3">Analysis Factors</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Historical disaster frequency</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Regional geographical factors</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Current policy effectiveness</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Population density and infrastructure</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Economic and social factors</span>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-3">Policy Categories</h4>
                  <div className="space-y-2">
                    {[
                      "Early Warning Systems",
                      "Infrastructure Development",
                      "Community Preparedness",
                      "Emergency Response",
                      "Recovery and Rehabilitation",
                      "Risk Reduction Measures",
                    ].map((category, index) => (
                      <Badge key={index} variant="outline" className="mr-2 mb-2">
                        {category}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results" className="space-y-6">
          {policyResults ? (
            <>
              <Alert className="border-green-200 bg-green-50">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <AlertDescription className="text-green-700">
                  Successfully generated {policyResults.length} evidence-based policy recommendations based on disaster
                  patterns and regional analysis.
                </AlertDescription>
              </Alert>

              <div className="space-y-6">
                {policyResults.map((policy, index) => (
                  <Card key={index} className="border-l-4 border-l-blue-500">
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between">
                        <span>
                          {policy.region} - {policy.disaster_type} Policy
                        </span>
                        <div className="flex space-x-2">
                          <Badge variant={policy.frequency >= 5 ? "destructive" : "secondary"}>
                            Frequency: {policy.frequency}/decade
                          </Badge>
                          <Badge variant="outline">{policy.expected_impact}</Badge>
                        </div>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2">Current Policies</h4>
                        <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                          {policy.current_policies.map((currentPolicy, idx) => (
                            <li key={idx}>{currentPolicy}</li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h4 className="font-medium text-gray-900 mb-2">Evidence-Based Recommendation</h4>
                        <p className="text-sm text-gray-700 bg-blue-50 p-3 rounded-lg">
                          {policy.evidence_based_recommendation}
                        </p>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <h5 className="font-medium text-gray-900 mb-2">Regional Factors</h5>
                          <div className="space-y-1">
                            {policy.factors_considered.map((factor, idx) => (
                              <Badge key={idx} variant="outline" className="text-xs mr-1 mb-1">
                                {factor}
                              </Badge>
                            ))}
                          </div>
                        </div>
                        <div>
                          <h5 className="font-medium text-gray-900 mb-2">Implementation</h5>
                          <div className="text-sm text-gray-600">
                            <div>Timeline: {policy.implementation_timeline}</div>
                            <div>Budget: {policy.budget_estimate}</div>
                          </div>
                        </div>
                        <div>
                          <h5 className="font-medium text-gray-900 mb-2">Expected Impact</h5>
                          <div className="text-sm text-green-600 font-medium">{policy.expected_impact}</div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </>
          ) : (
            <Card>
              <CardContent className="p-12 text-center">
                <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Policy Results</h3>
                <p className="text-gray-600">Generate policies to see evidence-based recommendations</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
