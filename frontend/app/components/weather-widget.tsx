"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Cloud, Sun, CloudRain, Wind, Droplets, Eye, Gauge, MapPin, Thermometer } from "lucide-react"
interface WeatherWidgetProps {
  backendConnected: boolean
  backendUrl: string
}

type WeatherData =
  | {
      success: true
      data: {
        city: string
        temperature: number
        condition: string
        humidity: number
        windSpeed: number
        visibility: number
        pressure: number
        uvIndex: number
        feels_like: number
        weather_description: string
      }
    }
  | {
      success: false
      error: string
    }

export default function WeatherWidget({ backendConnected, backendUrl }: WeatherWidgetProps) {
  const [city, setCity] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [weatherData, setWeatherData] = useState<WeatherData>({
    success: true,
    data: {
      city: "Mumbai",
      temperature: 32,
      condition: "Partly Cloudy",
      humidity: 78,
      windSpeed: 12,
      visibility: 8,
      pressure: 1013,
      uvIndex: 7,
      feels_like: 35,
      weather_description: "Partly cloudy with high humidity",
    },
  })
  const handleSearch = async () => {
    if (!city.trim()) return

    setIsLoading(true)
    try {
      if (backendConnected) {
        // Call the real backend API
        const response = await fetch(`${backendUrl}/weather/${encodeURIComponent(city)}`)
        const data = await response.json()

        if (data.success) {
          setWeatherData({
            success: true,
            data: {
              city: data.data.location,
              temperature: Math.round(data.data.temperature),
              condition: data.data.weather_description || "Clear",
              humidity: data.data.humidity,
              windSpeed: Math.round(data.data.wind_speed),
              visibility: 8, // Mock data
              pressure: Math.round(data.data.pressure),
              uvIndex: 7, // Mock data
              feels_like: Math.round(data.data.feels_like || data.data.temperature),
              weather_description: data.data.weather_description,
            },
          })
        } else {
          throw new Error(data.error?.message || "Failed to fetch weather data")
        }
      } else {
        // Simulate API call with demo data
        await new Promise((resolve) => setTimeout(resolve, 1000))

        setWeatherData({
          success: true,
          data: {
            city: city,
            temperature: Math.round(Math.random() * 15 + 20),
            condition: ["Sunny", "Partly Cloudy", "Cloudy", "Rainy"][Math.floor(Math.random() * 4)],
            humidity: Math.round(Math.random() * 40 + 40),
            windSpeed: Math.round(Math.random() * 20 + 5),
            visibility: Math.round(Math.random() * 5 + 5),
            pressure: Math.round(Math.random() * 50 + 1000),
            uvIndex: Math.round(Math.random() * 10 + 1),
            feels_like: Math.round(Math.random() * 15 + 22),
            weather_description: "Demo weather data",
          },
        })
      }
    } catch (error) {
      if (error instanceof Error) {
        console.error("Weather API error:", error.message);
      } else {
        console.error("Weather API error:", error);
      }
    
      setWeatherData({
        success: false,
        error: backendConnected
          ? "Unable to connect to weather service"
          : "Demo mode - limited functionality",
      });
    } finally {
      setIsLoading(false);
    }
  }

  const getWeatherIcon = (condition:string) => {
    if (condition?.toLowerCase().includes("rain")) return <CloudRain className="h-12 w-12 text-blue-500" />
    if (condition?.toLowerCase().includes("cloud")) return <Cloud className="h-12 w-12 text-gray-500" />
    return <Sun className="h-12 w-12 text-yellow-500" />
  }

  const getAQIColor = (aqi:number) => {
    if (aqi <= 50) return "text-green-600"
    if (aqi <= 100) return "text-yellow-600"
    if (aqi <= 150) return "text-orange-600"
    return "text-red-600"
  }

  return (
    <div className="space-y-6">
      <Card className="bg-gradient-to-r from-blue-50 to-green-50">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Cloud className="h-5 w-5 text-blue-600" />
            <span>Real-time Weather Information</span>
            <Badge variant="outline" className="bg-blue-100 text-blue-700">
              {backendConnected ? "Live Data" : "Demo Mode"}
            </Badge>
          </CardTitle>
          <CardDescription>
            {backendConnected
              ? "Get comprehensive weather data powered by OpenWeatherMap API"
              : "Weather service running in demo mode (Backend not connected)"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!backendConnected && (
            <Alert className="mb-4 border-yellow-200 bg-yellow-50">
              <AlertDescription className="text-yellow-700">
                Backend not connected. Weather data is simulated for demonstration purposes.
              </AlertDescription>
            </Alert>
          )}

          <div className="flex space-x-2 mb-6">
            <div className="flex-1 relative">
              <MapPin className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Enter city name (e.g., Mumbai, Delhi, London)"
                value={city}
                onChange={(e) => setCity(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSearch()}
                className="pl-10"
              />
            </div>
            <Button onClick={handleSearch} disabled={isLoading || !city.trim()}>
              {isLoading ? "Loading..." : "Search"}
            </Button>
          </div>

          {!weatherData.success && weatherData.error && (
            <Alert className="mb-6 border-red-200 bg-red-50">
              <AlertDescription className="text-red-700">{weatherData.error}</AlertDescription>
            </Alert>
          )}

          {weatherData.success && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Current Weather */}
              <Card className="bg-gradient-to-br from-blue-500 to-blue-600 text-white">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-2xl font-bold">{weatherData.data.city}</h3>
                      <p className="opacity-90">{weatherData.data.condition}</p>
                    </div>
                    {getWeatherIcon(weatherData.data.condition)}
                  </div>
                  <div className="text-4xl font-bold mb-2">{weatherData.data.temperature}°C</div>
                  <div className="flex items-center space-x-4 text-sm opacity-90">
                    <span className="flex items-center">
                      <Thermometer className="h-4 w-4 mr-1" />
                      Feels like {weatherData.data.feels_like}°C
                    </span>
                    <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
                      {backendConnected ? "Live Data" : "Demo"}
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              {/* Weather Details */}
              <div className="grid grid-cols-2 gap-4">
                <Card className="bg-gradient-to-br from-blue-50 to-blue-100">
                  <CardContent className="p-4 text-center">
                    <Droplets className="h-6 w-6 text-blue-500 mx-auto mb-2" />
                    <div className="text-2xl font-bold">{weatherData.data.humidity}%</div>
                    <div className="text-sm text-gray-600">Humidity</div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-gray-50 to-gray-100">
                  <CardContent className="p-4 text-center">
                    <Wind className="h-6 w-6 text-gray-500 mx-auto mb-2" />
                    <div className="text-2xl font-bold">{weatherData.data.windSpeed}</div>
                    <div className="text-sm text-gray-600">km/h Wind</div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-green-50 to-green-100">
                  <CardContent className="p-4 text-center">
                    <Eye className="h-6 w-6 text-green-500 mx-auto mb-2" />
                    <div className="text-2xl font-bold">{weatherData.data.visibility}</div>
                    <div className="text-sm text-gray-600">km Visibility</div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-purple-50 to-purple-100">
                  <CardContent className="p-4 text-center">
                    <Gauge className="h-6 w-6 text-purple-500 mx-auto mb-2" />
                    <div className="text-2xl font-bold">{weatherData.data.pressure}</div>
                    <div className="text-sm text-gray-600">hPa Pressure</div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 7-Day Forecast */}
      <Card>
        <CardHeader>
          <CardTitle>7-Day Weather Forecast</CardTitle>
          <CardDescription>Extended weather outlook {!backendConnected && "(Demo data)"}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-7 gap-4">
            {[
              { day: "Today", high: 32, low: 24, condition: "Sunny", precipitation: 0 },
              { day: "Tomorrow", high: 34, low: 26, condition: "Partly Cloudy", precipitation: 10 },
              { day: "Wed", high: 31, low: 23, condition: "Cloudy", precipitation: 30 },
              { day: "Thu", high: 29, low: 22, condition: "Rain", precipitation: 80 },
              { day: "Fri", high: 28, low: 21, condition: "Rain", precipitation: 90 },
              { day: "Sat", high: 30, low: 23, condition: "Partly Cloudy", precipitation: 20 },
              { day: "Sun", high: 33, low: 25, condition: "Sunny", precipitation: 0 },
            ].map((forecast, index) => (
              <Card key={index} className="text-center hover:shadow-md transition-shadow">
                <CardContent className="p-4">
                  <div className="font-medium mb-2">{forecast.day}</div>
                  {getWeatherIcon(forecast.condition)}
                  <div className="text-sm mt-2">
                    <div className="font-bold">{forecast.high}°</div>
                    <div className="text-gray-600">{forecast.low}°</div>
                  </div>
                  <div className="flex items-center justify-center mt-2">
                    <Droplets className="h-3 w-3 text-blue-500 mr-1" />
                    <span className="text-xs">{forecast.precipitation}%</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Weather Alerts */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Sun className="h-5 w-5 text-orange-600" />
            <span>Weather Alerts & Advisories</span>
            {!backendConnected && <Badge variant="outline">Demo</Badge>}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <Alert className="border-orange-200 bg-orange-50">
              <Sun className="h-4 w-4 text-orange-600" />
              <AlertDescription className="text-orange-700">
                <strong>Heat Advisory:</strong> High temperatures expected. Stay hydrated and avoid prolonged outdoor
                exposure during peak hours (11 AM - 4 PM).
              </AlertDescription>
            </Alert>

            <Alert className="border-blue-200 bg-blue-50">
              <CloudRain className="h-4 w-4 text-blue-600" />
              <AlertDescription className="text-blue-700">
                <strong>Monsoon Update:</strong> Moderate to heavy rainfall expected Thursday-Friday. Plan indoor
                activities and avoid flood-prone areas.
              </AlertDescription>
            </Alert>
          </div>
        </CardContent>
      </Card>

      {/* Air Quality Index */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Wind className="h-5 w-5 text-gray-600" />
            <span>Air Quality Index</span>
            {!backendConnected && <Badge variant="outline">Demo</Badge>}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className={`text-4xl font-bold ${getAQIColor(156)}`}>156</div>
              <div className="text-sm text-gray-600 mb-2">Current AQI</div>
              <Badge variant="destructive">Unhealthy</Badge>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>PM2.5:</span>
                <span className="font-medium">89 μg/m³</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>PM10:</span>
                <span className="font-medium">134 μg/m³</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>O3:</span>
                <span className="font-medium">67 μg/m³</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>NO2:</span>
                <span className="font-medium">45 μg/m³</span>
              </div>
            </div>
            <div className="text-sm text-gray-600">
              <p className="mb-2">
                <strong>Health Advisory:</strong>
              </p>
              <p>
                Sensitive groups should limit outdoor activities. Everyone should reduce prolonged outdoor exertion.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
