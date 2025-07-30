"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { FileText, MapPin, Camera, Send, CheckCircle, Clock, AlertTriangle } from "lucide-react"

interface CitizenReportingProps {
  backendConnected: boolean
  backendUrl: string
}

export default function CitizenReporting({ backendConnected, backendUrl }: CitizenReportingProps) {
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [reportForm, setReportForm] = useState({
    location: { lat: 0, lon: 0 },
    disaster_type: "",
    severity: 5,
    description: "",
    image_url: "",
  })
  const [reports, setReports] = useState([
    {
      id: "1",
      location: { lat: 19.076, lon: 72.8777 },
      disaster_type: "flood",
      severity: 8,
      description: "Heavy flooding in downtown area, roads are impassable",
      timestamp: "2024-01-15T10:30:00Z",
      verified: true,
    },
    {
      id: "2",
      location: { lat: 28.6139, lon: 77.209 },
      disaster_type: "heatwave",
      severity: 6,
      description: "Extreme heat conditions, multiple heat stroke cases reported",
      timestamp: "2024-01-15T08:15:00Z",
      verified: false,
    },
  ])

  const handleSubmit = async () => {
    if (!reportForm.disaster_type || !reportForm.description) {
      alert("Please fill in all required fields")
      return
    }

    setIsSubmitting(true)

    try {
      if (backendConnected) {
        // Call the real backend API
        const response = await fetch(`${backendUrl}/citizen-report`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(reportForm),
        })

        if (response.ok) {
          const data = await response.json()
          const newReport = {
            id: data.report_id,
            ...reportForm,
            timestamp: new Date().toISOString(),
            verified: data.verified,
          }
          setReports([newReport, ...reports])
        } else {
          throw new Error("Failed to submit report")
        }
      } else {
        // Simulate API call with demo data
        await new Promise((resolve) => setTimeout(resolve, 1500))

        const newReport = {
          id: Date.now().toString(),
          ...reportForm,
          timestamp: new Date().toISOString(),
          verified: Math.random() > 0.5,
        }
        setReports([newReport, ...reports])
      }

      // Reset form
      setReportForm({
        location: { lat: 0, lon: 0 },
        disaster_type: "",
        severity: 5,
        description: "",
        image_url: "",
      })
    } catch (error) {
      console.error("Report submission failed:", error)
      alert("Failed to submit report. Please try again.")
    } finally {
      setIsSubmitting(false)
    }
  }

  const getDisasterIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case "flood":
        return "🌊"
      case "earthquake":
        return "🏠"
      case "heatwave":
        return "🌡️"
      case "cyclone":
        return "🌪️"
      case "drought":
        return "🏜️"
      default:
        return "⚠️"
    }
  }

  const getSeverityColor = (severity: number) => {
    if (severity >= 8) return "text-red-600"
    if (severity >= 6) return "text-orange-600"
    if (severity >= 4) return "text-yellow-600"
    return "text-green-600"
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="bg-gradient-to-r from-green-50 to-blue-50">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <FileText className="h-6 w-6 text-green-600" />
            <span>Citizen Disaster Reporting</span>
            <Badge variant="outline" className="bg-green-100 text-green-700">
              {backendConnected ? "Blockchain Verified" : "Demo Mode"}
            </Badge>
          </CardTitle>
          <CardDescription>
            {backendConnected
              ? "Report disasters and emergencies directly to authorities with blockchain verification"
              : "Disaster reporting system running in demo mode (Backend not connected)"}
          </CardDescription>
        </CardHeader>
      </Card>

      {!backendConnected && (
        <Alert className="border-yellow-200 bg-yellow-50">
          <AlertTriangle className="h-4 w-4 text-yellow-600" />
          <AlertDescription className="text-yellow-700">
            Backend not connected. Reports will be simulated for demonstration purposes.
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Report Submission Form */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Send className="h-5 w-5 text-blue-600" />
              <span>Submit New Report</span>
            </CardTitle>
            <CardDescription>Report disasters and emergencies in your area</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Disaster Type *</label>
              <Select
                value={reportForm.disaster_type}
                onValueChange={(value) => setReportForm((prev) => ({ ...prev, disaster_type: value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select disaster type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="flood">🌊 Flood</SelectItem>
                  <SelectItem value="earthquake">🏠 Earthquake</SelectItem>
                  <SelectItem value="heatwave">🌡️ Heat Wave</SelectItem>
                  <SelectItem value="cyclone">🌪️ Cyclone</SelectItem>
                  <SelectItem value="drought">🏜️ Drought</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Severity (1-10) *</label>
              <Input
                type="number"
                min="1"
                max="10"
                value={reportForm.severity}
                onChange={(e) => setReportForm((prev) => ({ ...prev, severity: Number.parseInt(e.target.value) || 1 }))}
                placeholder="Rate severity from 1 (minor) to 10 (catastrophic)"
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Location</label>
              <div className="flex items-center space-x-2">
                <MapPin className="h-4 w-4 text-gray-400" />
                <Input placeholder="Current location will be auto-detected" value="Auto-detected location" disabled />
                <Button variant="outline" size="sm">
                  📍 Detect
                </Button>
              </div>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Description *</label>
              <Textarea
                placeholder="Describe the situation in detail..."
                value={reportForm.description}
                onChange={(e) => setReportForm((prev) => ({ ...prev, description: e.target.value }))}
                rows={4}
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Photo Evidence (Optional)</label>
              <div className="flex items-center space-x-2">
                <Camera className="h-4 w-4 text-gray-400" />
                <Input type="file" accept="image/*" placeholder="Upload photo" />
              </div>
            </div>

            <Button
              onClick={handleSubmit}
              disabled={isSubmitting || !reportForm.disaster_type || !reportForm.description}
              className="w-full"
            >
              {isSubmitting ? (
                <>
                  <Clock className="h-4 w-4 mr-2 animate-spin" />
                  {backendConnected ? "Submitting to Blockchain..." : "Submitting Report..."}
                </>
              ) : (
                <>
                  <Send className="h-4 w-4 mr-2" />
                  Submit Report
                </>
              )}
            </Button>

            {backendConnected && (
              <div className="text-xs text-gray-600 bg-blue-50 p-3 rounded-lg">
                <strong>Blockchain Security:</strong> Your report will be cryptographically signed and stored on the
                blockchain for transparency and immutability.
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Reports */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <FileText className="h-5 w-5 text-green-600" />
              <span>Recent Reports</span>
            </CardTitle>
            <CardDescription>Latest citizen-submitted disaster reports</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {reports.map((report) => (
                <div key={report.id} className="p-4 border rounded-lg hover:bg-gray-50 transition-colors">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-2xl">{getDisasterIcon(report.disaster_type)}</span>
                      <span className="font-medium capitalize">{report.disaster_type}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant={report.verified ? "default" : "secondary"}>
                        {report.verified ? (
                          <>
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Verified
                          </>
                        ) : (
                          <>
                            <Clock className="h-3 w-3 mr-1" />
                            Pending
                          </>
                        )}
                      </Badge>
                      <span className={`text-sm font-medium ${getSeverityColor(report.severity)}`}>
                        Severity: {report.severity}/10
                      </span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-700 mb-2">{report.description}</p>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span className="flex items-center">
                      <MapPin className="h-3 w-3 mr-1" />
                      {report.location.lat.toFixed(4)}, {report.location.lon.toFixed(4)}
                    </span>
                    <span>{new Date(report.timestamp).toLocaleString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Statistics */}
      <Card>
        <CardHeader>
          <CardTitle>Reporting Statistics</CardTitle>
          <CardDescription>Overview of citizen reporting activity</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{reports.length}</div>
              <div className="text-sm text-gray-600">Total Reports</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{reports.filter((r) => r.verified).length}</div>
              <div className="text-sm text-gray-600">Verified Reports</div>
            </div>
            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">{reports.filter((r) => !r.verified).length}</div>
              <div className="text-sm text-gray-600">Pending Verification</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {(reports.reduce((sum, r) => sum + r.severity, 0) / reports.length).toFixed(1)}
              </div>
              <div className="text-sm text-gray-600">Average Severity</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
