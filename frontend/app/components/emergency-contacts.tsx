"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Phone, MapPin, Clock, Search, AlertTriangle, Shield, Heart, Flame } from "lucide-react"

interface EmergencyContactsProps {
  backendConnected: boolean
  backendUrl: string
}

export default function EmergencyContacts({ backendConnected, backendUrl }: EmergencyContactsProps) {
  const [searchLocation, setSearchLocation] = useState("")
  const [selectedCategory, setSelectedCategory] = useState("all")

  const emergencyContacts = [
    {
      category: "police",
      name: "Police Emergency",
      number: "100",
      description: "For crimes, accidents, and general emergencies",
      icon: Shield,
      color: "text-blue-600",
      available: "24/7",
    },
    {
      category: "fire",
      name: "Fire Brigade",
      number: "101",
      description: "Fire emergencies, rescue operations",
      icon: Flame,
      color: "text-red-600",
      available: "24/7",
    },
    {
      category: "medical",
      name: "Ambulance",
      number: "108",
      description: "Medical emergencies, ambulance services",
      icon: Heart,
      color: "text-green-600",
      available: "24/7",
    },
    {
      category: "disaster",
      name: "Disaster Management",
      number: "1070",
      description: "Natural disasters, evacuation assistance",
      icon: AlertTriangle,
      color: "text-orange-600",
      available: "24/7",
    },
    {
      category: "medical",
      name: "Women Helpline",
      number: "1091",
      description: "Women safety and emergency assistance",
      icon: Shield,
      color: "text-purple-600",
      available: "24/7",
    },
    {
      category: "medical",
      name: "Child Helpline",
      number: "1098",
      description: "Child safety and emergency assistance",
      icon: Heart,
      color: "text-pink-600",
      available: "24/7",
    },
  ]

  const localContacts = [
    {
      name: "Mumbai Police Control Room",
      number: "+91-22-2262-0111",
      area: "Mumbai",
      type: "Police",
    },
    {
      name: "Delhi Fire Service",
      number: "+91-11-2338-1011",
      area: "Delhi",
      type: "Fire",
    },
    {
      name: "Bangalore Traffic Police",
      number: "+91-80-2294-2444",
      area: "Bangalore",
      type: "Traffic",
    },
  ]

  const filteredContacts = emergencyContacts.filter(
    (contact) => selectedCategory === "all" || contact.category === selectedCategory,
  )

  const makeCall = (number: string) => {
    if (typeof window !== "undefined") {
      window.location.href = `tel:${number}`
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="bg-gradient-to-r from-red-50 to-orange-50">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Phone className="h-6 w-6 text-red-600" />
            <span>Emergency Contacts</span>
            <Badge variant="outline" className="bg-red-100 text-red-700">
              24/7 Available
            </Badge>
          </CardTitle>
          <CardDescription>
            Quick access to emergency services and local contacts for immediate assistance
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Emergency Alert */}
      <Alert className="border-red-200 bg-red-50">
        <AlertTriangle className="h-4 w-4 text-red-600" />
        <AlertDescription className="text-red-700">
          <strong>In case of immediate emergency:</strong> Call 112 (National Emergency Number) for instant connection
          to police, fire, and medical services.
        </AlertDescription>
      </Alert>

      {/* Search and Filter */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search by location..."
                value={searchLocation}
                onChange={(e) => setSearchLocation(e.target.value)}
                className="pl-10"
              />
            </div>
            <div className="flex gap-2">
              {[
                { id: "all", label: "All", icon: "🚨" },
                { id: "police", label: "Police", icon: "👮" },
                { id: "fire", label: "Fire", icon: "🚒" },
                { id: "medical", label: "Medical", icon: "🚑" },
                { id: "disaster", label: "Disaster", icon: "🌪️" },
              ].map((category) => (
                <Button
                  key={category.id}
                  variant={selectedCategory === category.id ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSelectedCategory(category.id)}
                  className="whitespace-nowrap"
                >
                  {category.icon} {category.label}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* National Emergency Numbers */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Phone className="h-5 w-5 text-red-600" />
            <span>National Emergency Numbers</span>
          </CardTitle>
          <CardDescription>India's primary emergency contact numbers</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredContacts.map((contact, index) => (
              <Card
                key={index}
                className="hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => makeCall(contact.number)}
              >
                <CardContent className="p-4">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className={`p-2 rounded-full bg-gray-100`}>
                      <contact.icon className={`h-5 w-5 ${contact.color}`} />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-lg">{contact.number}</h3>
                      <p className="text-sm font-medium text-gray-900">{contact.name}</p>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mb-3">{contact.description}</p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-1">
                      <Clock className="h-3 w-3 text-green-600" />
                      <span className="text-xs text-green-600 font-medium">{contact.available}</span>
                    </div>
                    <Button size="sm" className="bg-red-600 hover:bg-red-700">
                      <Phone className="h-3 w-3 mr-1" />
                      Call
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Local Emergency Contacts */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <MapPin className="h-5 w-5 text-blue-600" />
            <span>Local Emergency Contacts</span>
          </CardTitle>
          <CardDescription>Region-specific emergency services and contacts</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {localContacts
              .filter(
                (contact) =>
                  searchLocation === "" ||
                  contact.area.toLowerCase().includes(searchLocation.toLowerCase()) ||
                  contact.name.toLowerCase().includes(searchLocation.toLowerCase()),
              )
              .map((contact, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center space-x-2">
                        <MapPin className="h-4 w-4 text-gray-500" />
                        <span className="text-sm font-medium text-gray-600">{contact.area}</span>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {contact.type}
                      </Badge>
                    </div>
                    <h4 className="font-medium text-gray-900 mt-1">{contact.name}</h4>
                    <p className="text-sm text-gray-600">{contact.number}</p>
                  </div>
                  <Button variant="outline" size="sm" onClick={() => makeCall(contact.number)}>
                    <Phone className="h-3 w-3 mr-1" />
                    Call
                  </Button>
                </div>
              ))}
          </div>
        </CardContent>
      </Card>

      {/* Emergency Preparedness Tips */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5 text-orange-600" />
            <span>Emergency Preparedness</span>
          </CardTitle>
          <CardDescription>Important tips for emergency situations</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3 text-red-700">Before Calling Emergency Services:</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start space-x-2">
                  <span className="text-red-600 mt-1">•</span>
                  <span>Stay calm and speak clearly</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-red-600 mt-1">•</span>
                  <span>Know your exact location</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-red-600 mt-1">•</span>
                  <span>Describe the emergency clearly</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-red-600 mt-1">•</span>
                  <span>Follow the operator's instructions</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-red-600 mt-1">•</span>
                  <span>Don't hang up until told to do so</span>
                </li>
              </ul>
            </div>

            <div>
              <h4 className="font-medium mb-3 text-blue-700">Emergency Kit Essentials:</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start space-x-2">
                  <span className="text-blue-600 mt-1">•</span>
                  <span>First aid supplies</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-blue-600 mt-1">•</span>
                  <span>Flashlight and batteries</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-blue-600 mt-1">•</span>
                  <span>Emergency contact list</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-blue-600 mt-1">•</span>
                  <span>Important documents (copies)</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-blue-600 mt-1">•</span>
                  <span>Water and non-perishable food</span>
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Emergency Actions</CardTitle>
          <CardDescription>One-tap access to emergency services</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Button className="h-20 bg-red-600 hover:bg-red-700 text-white flex-col" onClick={() => makeCall("112")}>
              <Phone className="h-6 w-6 mb-1" />
              <span className="text-sm">Emergency</span>
              <span className="text-xs opacity-90">112</span>
            </Button>

            <Button className="h-20 bg-blue-600 hover:bg-blue-700 text-white flex-col" onClick={() => makeCall("100")}>
              <Shield className="h-6 w-6 mb-1" />
              <span className="text-sm">Police</span>
              <span className="text-xs opacity-90">100</span>
            </Button>

            <Button className="h-20 bg-red-500 hover:bg-red-600 text-white flex-col" onClick={() => makeCall("101")}>
              <Flame className="h-6 w-6 mb-1" />
              <span className="text-sm">Fire</span>
              <span className="text-xs opacity-90">101</span>
            </Button>

            <Button
              className="h-20 bg-green-600 hover:bg-green-700 text-white flex-col"
              onClick={() => makeCall("108")}
            >
              <Heart className="h-6 w-6 mb-1" />
              <span className="text-sm">Medical</span>
              <span className="text-xs opacity-90">108</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
