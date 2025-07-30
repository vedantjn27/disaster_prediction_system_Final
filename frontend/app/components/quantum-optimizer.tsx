"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Zap, Activity, BarChart3, TrendingUp, Cpu, Database, CheckCircle } from "lucide-react"

interface QuantumOptimizerProps {
  backendConnected: boolean
  backendUrl: string
}

export default function QuantumOptimizer({ backendConnected, backendUrl }: QuantumOptimizerProps) {
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [optimizationResult, setOptimizationResult] = useState(null)
  const [regions, setRegions] = useState(["Delhi", "Mumbai", "Bangalore"])
  const [resources, setResources] = useState({
    medical_supplies: 1000,
    emergency_personnel: 500,
    vehicles: 200,
    shelter_capacity: 2000,
  })
  const [demands, setDemands] = useState({
    Delhi: { medical_supplies: 400, emergency_personnel: 200, vehicles: 80, shelter_capacity: 800 },
    Mumbai: { medical_supplies: 350, emergency_personnel: 150, vehicles: 60, shelter_capacity: 600 },
    Bangalore: { medical_supplies: 250, emergency_personnel: 100, vehicles: 40, shelter_capacity: 400 },
  })

  const [newRegion, setNewRegion] = useState("")
  const [selectedResource, setSelectedResource] = useState("")
  const [resourceValue, setResourceValue] = useState("")

  const handleOptimize = async () => {
    setIsOptimizing(true)

    try {
      if (backendConnected) {
        // Call the real backend API
        const response = await fetch(`${backendUrl}/optimize-resources`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            regions,
            resources,
            demands,
          }),
        })

        if (response.ok) {
          const data = await response.json()
          setOptimizationResult(data)
        } else {
          throw new Error("Optimization failed")
        }
      } else {
        // Simulate quantum optimization with demo data
        await new Promise((resolve) => setTimeout(resolve, 3000))

        const totalDemand = Object.values(demands).reduce((sum, regionDemand) => {
          return sum + Object.values(regionDemand).reduce((a, b) => a + b, 0)
        }, 0)
        const totalSupply = Object.values(resources).reduce((a, b) => a + b, 0)

        const mockResult = {
          status: totalSupply >= totalDemand ? "optimal" : "suboptimal",
          total_supply: totalSupply,
          total_demand: totalDemand,
          allocation: {},
          efficiency_score: Math.min((totalSupply / totalDemand) * 100, 100),
          quantum_runtime: Math.random() * 2 + 1,
        }

        // Distribute resources proportionally
        regions.forEach((region) => {
          const regionDemand = demands[region] || {}
          const regionTotal = Object.values(regionDemand).reduce((a, b) => a + b, 0)
          const allocationRatio = Math.min(totalSupply / totalDemand, 1.0)

          mockResult.allocation[region] = {}
          Object.entries(regionDemand).forEach(([resource, demand]) => {
            mockResult.allocation[region][resource] = Math.floor(demand * allocationRatio)
          })
        })

        setOptimizationResult(mockResult)
      }
    } catch (error) {
      console.error("Optimization failed:", error)
      alert("Optimization failed. Please try again.")
    } finally {
      setIsOptimizing(false)
    }
  }

  const addRegion = () => {
    if (newRegion && !regions.includes(newRegion)) {
      setRegions([...regions, newRegion])
      setDemands({
        ...demands,
        [newRegion]: { medical_supplies: 100, emergency_personnel: 50, vehicles: 20, shelter_capacity: 200 },
      })
      setNewRegion("")
    }
  }

  const updateResource = () => {
    if (selectedResource && resourceValue) {
      setResources({
        ...resources,
        [selectedResource]: Number.parseInt(resourceValue),
      })
      setSelectedResource("")
      setResourceValue("")
    }
  }

  const updateDemand = (region: string, resource: string, value: string) => {
    setDemands({
      ...demands,
      [region]: {
        ...demands[region],
        [resource]: Number.parseInt(value) || 0,
      },
    })
  }

  return (
    <div className="space-y-6">
      {/* Quantum Optimizer Header */}
      <Card className="bg-gradient-to-r from-purple-50 to-blue-50">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Zap className="h-6 w-6 text-purple-600" />
            <span>Quantum Resource Optimizer</span>
            <Badge variant="outline" className="bg-purple-100 text-purple-700">
              {backendConnected ? "QAOA Algorithm" : "Demo Mode"}
            </Badge>
          </CardTitle>
          <CardDescription>
            {backendConnected
              ? "Optimize resource allocation using Quantum Approximate Optimization Algorithm (QAOA)"
              : "Simulated quantum optimization for resource allocation (Backend not connected)"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-white rounded-lg">
              <Cpu className="h-6 w-6 text-purple-500 mx-auto mb-2" />
              <div className="text-lg font-bold">{regions.length}</div>
              <div className="text-sm text-gray-600">Regions</div>
            </div>
            <div className="text-center p-3 bg-white rounded-lg">
              <Database className="h-6 w-6 text-blue-500 mx-auto mb-2" />
              <div className="text-lg font-bold">{Object.keys(resources).length}</div>
              <div className="text-sm text-gray-600">Resource Types</div>
            </div>
            <div className="text-center p-3 bg-white rounded-lg">
              <Activity className="h-6 w-6 text-green-500 mx-auto mb-2" />
              <div className="text-lg font-bold">{optimizationResult?.efficiency_score?.toFixed(1) || 0}%</div>
              <div className="text-sm text-gray-600">Efficiency</div>
            </div>
            <div className="text-center p-3 bg-white rounded-lg">
              <TrendingUp className="h-6 w-6 text-orange-500 mx-auto mb-2" />
              <div className="text-lg font-bold">{optimizationResult?.quantum_runtime?.toFixed(2) || 0}s</div>
              <div className="text-sm text-gray-600">Runtime</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {!backendConnected && (
        <Alert className="border-yellow-200 bg-yellow-50">
          <Zap className="h-4 w-4 text-yellow-600" />
          <AlertDescription className="text-yellow-700">
            Backend not connected. Running in demo mode with simulated quantum optimization.
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="setup" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="setup">Setup & Configuration</TabsTrigger>
          <TabsTrigger value="optimize">Quantum Optimization</TabsTrigger>
          <TabsTrigger value="results">Results & Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="setup" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Resource Configuration */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Database className="h-5 w-5 text-blue-600" />
                  <span>Available Resources</span>
                </CardTitle>
                <CardDescription>Configure total available resources for optimization</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  {Object.entries(resources).map(([resource, value]) => (
                    <div key={resource} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <span className="font-medium capitalize">{resource.replace("_", " ")}</span>
                      <Badge variant="outline">{value}</Badge>
                    </div>
                  ))}
                </div>

                <div className="border-t pt-4">
                  <h4 className="font-medium mb-3">Update Resource</h4>
                  <div className="flex space-x-2">
                    <Select value={selectedResource} onValueChange={setSelectedResource}>
                      <SelectTrigger className="flex-1">
                        <SelectValue placeholder="Select resource" />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.keys(resources).map((resource) => (
                          <SelectItem key={resource} value={resource}>
                            {resource.replace("_", " ")}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Input
                      type="number"
                      placeholder="Value"
                      value={resourceValue}
                      onChange={(e) => setResourceValue(e.target.value)}
                      className="w-24"
                    />
                    <Button onClick={updateResource}>Update</Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Region Management */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="h-5 w-5 text-green-600" />
                  <span>Regions & Demands</span>
                </CardTitle>
                <CardDescription>Manage regions and their resource demands</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  {regions.map((region) => (
                    <div key={region} className="p-3 bg-gray-50 rounded-lg">
                      <div className="font-medium mb-2">{region}</div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        {Object.entries(demands[region] || {}).map(([resource, demand]) => (
                          <div key={resource} className="flex justify-between">
                            <span>{resource.replace("_", " ")}:</span>
                            <input
                              type="number"
                              value={demand}
                              onChange={(e) => updateDemand(region, resource, e.target.value)}
                              className="w-16 px-1 border rounded text-right"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="border-t pt-4">
                  <h4 className="font-medium mb-3">Add Region</h4>
                  <div className="flex space-x-2">
                    <Input placeholder="Region name" value={newRegion} onChange={(e) => setNewRegion(e.target.value)} />
                    <Button onClick={addRegion}>Add</Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="optimize" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Zap className="h-5 w-5 text-purple-600" />
                <span>Quantum Optimization Engine</span>
              </CardTitle>
              <CardDescription>
                {backendConnected
                  ? "Run QAOA algorithm to find optimal resource allocation"
                  : "Simulate quantum optimization algorithm"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="text-center space-y-4">
                <Button
                  onClick={handleOptimize}
                  disabled={isOptimizing}
                  size="lg"
                  className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
                >
                  {isOptimizing ? (
                    <>
                      <Activity className="h-4 w-4 mr-2 animate-spin" />
                      {backendConnected ? "Running Quantum Optimization..." : "Simulating Optimization..."}
                    </>
                  ) : (
                    <>
                      <Zap className="h-4 w-4 mr-2" />
                      {backendConnected ? "Run Quantum Optimization" : "Run Simulation"}
                    </>
                  )}
                </Button>

                {isOptimizing && (
                  <div className="space-y-3">
                    <Progress value={33} className="h-2" />
                    <Alert className="border-purple-200 bg-purple-50">
                      <Cpu className="h-4 w-4 text-purple-600" />
                      <AlertDescription className="text-purple-700">
                        {backendConnected
                          ? "Quantum circuits are processing resource allocation optimization using QAOA algorithm..."
                          : "Simulating quantum optimization process with classical algorithms..."}
                      </AlertDescription>
                    </Alert>
                  </div>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium mb-3">Optimization Process</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Initialize quantum circuits</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Encode resource constraints</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Apply QAOA variational layers</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Activity
                        className={`h-4 w-4 ${isOptimizing ? "text-blue-500 animate-spin" : "text-gray-400"}`}
                      />
                      <span>Optimize allocation parameters</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Activity className={`h-4 w-4 ${isOptimizing ? "text-blue-500" : "text-gray-400"}`} />
                      <span>Extract optimal solution</span>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-3">Algorithm Details</h4>
                  <div className="space-y-2 text-sm text-gray-600">
                    <div>
                      <strong>Algorithm:</strong>{" "}
                      {backendConnected ? "QAOA (Quantum Approximate Optimization)" : "Classical Simulation"}
                    </div>
                    <div>
                      <strong>Objective:</strong> Minimize resource waste while meeting demands
                    </div>
                    <div>
                      <strong>Constraints:</strong> Regional demand satisfaction
                    </div>
                    <div>
                      <strong>Variables:</strong> {regions.length} regions × {Object.keys(resources).length} resources
                    </div>
                    <div>
                      <strong>Quantum Advantage:</strong>{" "}
                      {backendConnected ? "Exponential speedup for large problems" : "Simulated quantum behavior"}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results" className="space-y-6">
          {optimizationResult ? (
            <>
              {/* Optimization Summary */}
              <Card className="bg-gradient-to-r from-white to-gray-50">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Optimization Results</span>
                    <Badge variant={optimizationResult.status === "optimal" ? "default" : "secondary"}>
                      {optimizationResult.status}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-blue-600">
                        {optimizationResult.efficiency_score.toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-600">Efficiency Score</div>
                      <Progress value={optimizationResult.efficiency_score} className="mt-2" />
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">{optimizationResult.total_supply}</div>
                      <div className="text-sm text-gray-600">Total Supply</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600">{optimizationResult.total_demand}</div>
                      <div className="text-sm text-gray-600">Total Demand</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {optimizationResult.quantum_runtime.toFixed(2)}s
                      </div>
                      <div className="text-sm text-gray-600">Runtime</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Resource Allocation */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <BarChart3 className="h-5 w-5 text-green-600" />
                    <span>Optimal Resource Allocation</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {Object.entries(optimizationResult.allocation).map(([region, allocation]) => (
                      <div key={region} className="p-4 bg-gray-50 rounded-lg">
                        <h4 className="font-medium mb-3">{region}</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {Object.entries(allocation).map(([resource, amount]) => (
                            <div key={resource} className="text-center p-2 bg-white rounded">
                              <div className="text-lg font-bold">{amount}</div>
                              <div className="text-xs text-gray-600">{resource.replace("_", " ")}</div>
                              <div className="text-xs text-green-600">
                                {demands[region] && demands[region][resource]
                                  ? `${Math.round((amount / demands[region][resource]) * 100)}% of demand`
                                  : "N/A"}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Performance Metrics */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <TrendingUp className="h-5 w-5 text-blue-600" />
                    <span>Performance Analysis</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">
                        {optimizationResult.status === "optimal" ? "100%" : "85%"}
                      </div>
                      <div className="text-sm text-gray-600">Demand Satisfaction</div>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <div className="text-2xl font-bold text-green-600">
                        {Math.max(0, optimizationResult.total_supply - optimizationResult.total_demand)}
                      </div>
                      <div className="text-sm text-gray-600">Resource Surplus</div>
                    </div>
                    <div className="text-center p-4 bg-purple-50 rounded-lg">
                      <div className="text-2xl font-bold text-purple-600">
                        {backendConnected ? "Quantum" : "Classical"}
                      </div>
                      <div className="text-sm text-gray-600">Algorithm Type</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent className="p-12 text-center">
                <Zap className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Optimization Results</h3>
                <p className="text-gray-600">Run the quantum optimization to see results</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
