"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { BarChart3, TrendingUp, Users, Building, AlertTriangle, MessageSquare } from "lucide-react"

interface BlockchainAnalyticsProps {
  backendConnected: boolean
  backendUrl: string
}

export default function BlockchainAnalytics({ backendConnected, backendUrl }: BlockchainAnalyticsProps) {
  const analyticsData = {
    totalTransactions: 1247,
    registeredCitizens: 892,
    verifiedOrganizations: 23,
    activeAlerts: 7,
    feedbackSubmissions: 156,
    governmentActions: 89,
    blockchainIntegrity: 100,
    averageBlockTime: 12.5,
    networkHashRate: "2.4 TH/s",
    consensusEfficiency: 98.7,
  }

  const transactionTypes = [
    { type: "Citizen Registration", count: 892, percentage: 71.5, color: "bg-blue-500" },
    { type: "Feedback Submissions", count: 156, percentage: 12.5, color: "bg-green-500" },
    { type: "Government Actions", count: 89, percentage: 7.1, color: "bg-purple-500" },
    { type: "Alert Submissions", count: 67, percentage: 5.4, color: "bg-orange-500" },
    { type: "Organization Registration", count: 23, percentage: 1.8, color: "bg-red-500" },
    { type: "Other", count: 20, percentage: 1.6, color: "bg-gray-500" },
  ]

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6 text-center">
            <Users className="h-8 w-8 text-blue-500 mx-auto mb-2" />
            <div className="text-2xl font-bold">{analyticsData.registeredCitizens}</div>
            <div className="text-sm text-gray-600">Registered Citizens</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 text-center">
            <Building className="h-8 w-8 text-green-500 mx-auto mb-2" />
            <div className="text-2xl font-bold">{analyticsData.verifiedOrganizations}</div>
            <div className="text-sm text-gray-600">Verified Organizations</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 text-center">
            <AlertTriangle className="h-8 w-8 text-orange-500 mx-auto mb-2" />
            <div className="text-2xl font-bold">{analyticsData.activeAlerts}</div>
            <div className="text-sm text-gray-600">Active Alerts</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 text-center">
            <MessageSquare className="h-8 w-8 text-purple-500 mx-auto mb-2" />
            <div className="text-2xl font-bold">{analyticsData.feedbackSubmissions}</div>
            <div className="text-sm text-gray-600">Feedback Submissions</div>
          </CardContent>
        </Card>
      </div>

      {/* Transaction Distribution */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5 text-blue-600" />
            <span>Transaction Distribution</span>
            {!backendConnected && <Badge variant="outline">Demo Data</Badge>}
          </CardTitle>
          <CardDescription>Breakdown of transaction types on the blockchain</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {transactionTypes.map((transaction, index) => (
              <div key={index} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">{transaction.type}</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-600">{transaction.count}</span>
                    <Badge variant="outline">{transaction.percentage}%</Badge>
                  </div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${transaction.color}`}
                    style={{ width: `${transaction.percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Network Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-green-600" />
              <span>Network Performance</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm">Blockchain Integrity</span>
                  <span className="text-sm font-medium">{analyticsData.blockchainIntegrity}%</span>
                </div>
                <Progress value={analyticsData.blockchainIntegrity} className="h-2" />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm">Consensus Efficiency</span>
                  <span className="text-sm font-medium">{analyticsData.consensusEfficiency}%</span>
                </div>
                <Progress value={analyticsData.consensusEfficiency} className="h-2" />
              </div>

              <div className="grid grid-cols-2 gap-4 pt-2">
                <div className="text-center p-3 bg-gray-50 rounded">
                  <div className="text-lg font-bold text-blue-600">{analyticsData.averageBlockTime}s</div>
                  <div className="text-xs text-gray-600">Avg Block Time</div>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded">
                  <div className="text-lg font-bold text-green-600">{analyticsData.networkHashRate}</div>
                  <div className="text-xs text-gray-600">Hash Rate</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Transparency Metrics</CardTitle>
            <CardDescription>Government accountability through blockchain</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-blue-50 rounded">
                <span className="text-sm font-medium">Feedback Response Rate</span>
                <Badge variant="default">87%</Badge>
              </div>
              <div className="flex justify-between items-center p-3 bg-green-50 rounded">
                <span className="text-sm font-medium">Alert Verification Time</span>
                <Badge variant="default">&lt; 5 min</Badge>
              </div>
              <div className="flex justify-between items-center p-3 bg-purple-50 rounded">
                <span className="text-sm font-medium">Government Actions Logged</span>
                <Badge variant="default">{analyticsData.governmentActions}</Badge>
              </div>
              <div className="flex justify-between items-center p-3 bg-orange-50 rounded">
                <span className="text-sm font-medium">Public Audit Trail</span>
                <Badge variant="default">100% Accessible</Badge>
              </div>
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="text-sm font-medium">Backend Status</span>
                <Badge variant={backendConnected ? "default" : "secondary"}>
                  {backendConnected ? "Connected" : "Demo Mode"}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
