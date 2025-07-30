
"""
ClimaX: AI + Quantum + Blockchain-Powered Climate Resilience OS
Backend Implementation with RAG, Agentic AI, Quantum Simulation, and Blockchain 
"""

from fastapi import FastAPI, HTTPException, Depends,File, UploadFile, Form
from fastapi.responses import JSONResponse,FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from typing import List, Dict, Optional, Tuple, Union,Any
import requests
import asyncio
import json
import hashlib
import time
from datetime import datetime
import numpy as np
from dataclasses import dataclass,asdict
import logging
from enum import Enum
import uuid
import os
import aiohttp
from pathlib import Path
from dotenv import load_dotenv
import csv
import google.generativeai as genai
from transformers import pipeline
import pandas as pd
from twilio.rest import Client
import random
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import base64
from cryptography.exceptions import InvalidSignature
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from concurrent.futures import ThreadPoolExecutor
import pyttsx3
import threading
from queue import Queue
import tempfile
import speech_recognition as sr
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from fastapi.staticfiles import StaticFiles
import platform


citizen_public_keys = {}  # Maps citizen_id -> public key PEM string
# Create a directory for serving audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

load_dotenv()
# In-memory database to store citizens
citizens_db: Dict[str, Dict] = {}
organizations_db: Dict[str, Dict] = {}
alerts_db: List[Dict] = []
feedback_db = []  # Stores all submitted feedbacks
gov_actions_db = []  # list to store all logged government actions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ClimaX Backend", description="Climate Resilience OS Backend")

# Mount the audio directory to serve static files
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==================== MODELS ====================

class DisasterType(str, Enum):
    FLOOD = "flood"
    HEATWAVE = "heatwave"
    CYCLONE = "cyclone"
    DROUGHT = "drought"
    EARTHQUAKE = "earthquake"

class AlertLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class WeatherData:
    """Weather data structure"""
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    precipitation: float
    location: str
    timestamp: datetime
    weather_description: Optional[str] = None
    feels_like: Optional[float] = None


class CitizenReport(BaseModel):
    id: str = ""
    location: Dict[str, float]  # {"lat": float, "lon": float}
    disaster_type: DisasterType
    severity: int  # 1-10 scale
    description: str
    image_url: Optional[str] = None
    timestamp: Optional[datetime] = None
    verified: bool = False

class DisasterAlert(BaseModel):
    id: str
    region: str
    disaster_type: DisasterType
    alert_level: AlertLevel
    description: str
    affected_area: Dict[str, Any]
    evacuation_routes: List[Dict[str, Any]]
    resources_needed: Dict[str, int]
    timestamp: datetime
    blockchain_hash: str

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ResourceOptimization(BaseModel):
    region: str
    resources: Dict[str, int]
    demand: Dict[str, int]
    optimization_result: Dict[str, Any]
    quantum_runtime: float

class ChatRequest(BaseModel):
    question: str
    language: str = "en"
    location: Optional[str] = None

class WeatherRequest(BaseModel):
    location: str

class FeedbackRequest(BaseModel):
    safety_status: str
    govt_rating: float
    feedback: str
    language: str = "en"
    location: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    

class WeatherResponse(BaseModel):
    weather_info: str
    success: bool

class FeedbackResponse(BaseModel):
    success: bool
    message: str

class AnalysisRequest(BaseModel):
    region: str
    analysis_type: str

class ReportRequest(BaseModel):
    region: str
    report_type: str
    time_period: str

class StrategyRequest(BaseModel):
    region: str
    sector: str

class RegionData(BaseModel):
    climate_type: str
    major_issues: List[str]
    population: str
    key_sectors: List[str]
    current_aqi: int
    avg_temp: int
    rainfall_deficit: int
    renewable_percent: float

class ClimateMetrics(BaseModel):
    temperature: int
    aqi: int
    rainfall_deficit: int
    renewable_percent: float

class TrendData(BaseModel):
    dates: List[str]
    temperatures: List[float]
    aqi_values: List[float]

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speed: int = 150


@dataclass
class Organization:
    """Represents a federated organization"""
    id: str
    name: str
    type: str  # 'climate', 'health', 'emergency'
    public_key: str
    private_key: str = None  # Only stored for simulation
    
@dataclass
class Citizen:
    """Represents a citizen with privacy protection"""
    real_id: str
    hashed_id: str
    public_key: str
    private_key: str = None  # Only stored for simulation

@dataclass
class Transaction:
    """Base transaction structure"""
    tx_id: str
    timestamp: float
    tx_type: str
    data: Dict
    signatures: Dict[str, str]
    hash: str = None

@dataclass
class Block:
    """Blockchain block structure"""
    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    

class CitizenRegistration(BaseModel):
    citizen_id: str
    name: str
class OrganizationRegistration(BaseModel):
    organization_name: str

# Alert submission model
class AlertSubmission(BaseModel):
    organization_name: str
    alert_message: str
    alert_type: str
    affected_area: str
    timestamp: str  # ISO format string
    signature: str  # Base64 encoded

# Verification input model
class AlertVerificationRequest(BaseModel):
    organization_name: str
    alert_message: str
    alert_type: str
    affected_area: str
    timestamp: str
    signature: str

class Feedback(BaseModel):
    citizen_id: Optional[str]  # None for anonymous
    feedback: str
    signature: Optional[str]  # None for anonymous

class FeedbackVerify(BaseModel):
    feedback: str
    signature: str
    citizen_id: str

class GovAction(BaseModel):
    feedback_id: int
    action_taken: str
    officer_name: str
    timestamp: datetime = datetime.now()

class NewBlockData(BaseModel):
   data: List[Dict[str, str]]

class TamperRequest(BaseModel):
    block_index: int
    new_data: str

# ==================== QUANTUM SIMULATION MODULE ====================
class QuantumResourceOptimizer:
    """Simulates Quantum Approximate Optimization Algorithm (QAOA) for resource allocation"""

    def __init__(self):
        self.name = "Quantum Resource Optimizer"
        logger.info("Initialized Quantum Resource Optimizer (Simulated)")

    async def optimize_resources(self, regions: List[str], resources: Dict[str, int],
                                demands: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """
        Simulates quantum optimization for multi-region resource allocation
        In real implementation, this would use Qiskit with QAOA
        """
        start_time = time.time()

        # Simulate quantum computation delay
        await asyncio.sleep(0.1)

        # Simulate optimization results
        total_demand = sum(sum(region_demand.values()) for region_demand in demands.values())
        total_supply = sum(resources.values())

        optimization_result = {
            "status": "optimal" if total_supply >= total_demand else "suboptimal",
            "total_supply": total_supply,
            "total_demand": total_demand,
            "allocation": {},
            "efficiency_score": min(total_supply / total_demand, 1.0) * 100
        }

        # Distribute resources proportionally
        for region in regions:
            region_demand = demands.get(region, {})
            region_total = sum(region_demand.values())
            allocation_ratio = min(total_supply / total_demand, 1.0) if total_demand > 0 else 1.0

            optimization_result["allocation"][region] = {
                resource: int(demand * allocation_ratio)
                for resource, demand in region_demand.items()
            }

        runtime = time.time() - start_time
        optimization_result["quantum_runtime"] = runtime

        logger.info(f"Quantum optimization completed in {runtime:.3f}s")
        return optimization_result

class LocationAnalysisRequest(BaseModel):
    location: str = Field(..., description="Location name to analyze", min_length=1)
    gemini_api_key: str = Field(..., description="Google Gemini API key")

class LocationInfo(BaseModel):
    full_name: str
    country: str
    population: str
    area_km2: str
    coordinates: Dict[str, float]
    key_characteristics: List[str]

class CategoryScores(BaseModel):
    infrastructure: float = Field(..., ge=0, le=100)
    economic: float = Field(..., ge=0, le=100)
    social: float = Field(..., ge=0, le=100)
    environmental: float = Field(..., ge=0, le=100)
    governance: float = Field(..., ge=0, le=100)
    emergency: float = Field(..., ge=0, le=100)

class RiskAssessment(BaseModel):
    primary_threats: List[str]
    risk_level: str
    most_vulnerable_areas: List[str]
    climate_risks: List[str]

class Recommendations(BaseModel):
    immediate_actions: List[str]
    medium_term_improvements: List[str]
    long_term_strategic_goals: List[str]

class AnalysisResponse(BaseModel):
    location_info: LocationInfo
    category_scores: CategoryScores
    overall_resilience_score: float = Field(..., ge=0, le=100)
    risk_assessment: RiskAssessment
    strengths: List[str]
    vulnerabilities: List[str]
    recommendations: Recommendations
    data_sources_considered: List[str]
    confidence_level: str
    last_updated: str
    timestamp: str
    location_query: str

class PlotlyChart(BaseModel):
    data: List[Dict]
    layout: Dict

class DashboardResponse(BaseModel):
    gauge_chart: PlotlyChart
    radar_chart: PlotlyChart
    bar_chart: PlotlyChart
    comparison_chart: PlotlyChart

class HistoryItem(BaseModel):
    location_name: str
    overall_score: float
    risk_level: str
    timestamp: str

class HistoryResponse(BaseModel):
    analyses: List[HistoryItem]
    total_count: int

class StatusResponse(BaseModel):
    status: str
    message: str
    timestamp: str
class DisasterPattern(BaseModel):
    region: str
    disaster_type: str
    frequency: int

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speed: int = 150

class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_url: Optional[str] = None

class STTResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# ============================================================================
# CRYPTOGRAPHIC UTILITIES
# ============================================================================
class CryptoUtils:
    """Cryptographic utilities for the system"""
    
    @staticmethod
    def generate_keypair() -> Tuple[str, str]:
        """Generate RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        return public_pem, private_pem
    
    @staticmethod
    def sign_data(data: str, private_key_pem: str) -> str:
        """Sign data with private key"""
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode('utf-8'),
            password=None
        )
        
        signature = private_key.sign(
            data.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode('utf-8')
    
    @staticmethod
    def verify_signature(data: str, signature: str, public_key_pem: str) -> bool:
        """Verify signature with public key"""
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode('utf-8')
            )
            
            public_key.verify(
                base64.b64decode(signature.encode('utf-8')),
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    @staticmethod
    def hash_data(data: str) -> str:
        """Generate SHA-256 hash"""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_citizen_id(real_id: str, salt: str = "governance_salt") -> str:
        """Hash citizen ID for privacy"""
        return hashlib.sha256((real_id + salt).encode('utf-8')).hexdigest()
    
# ==================== BLOCKCHAIN SIMULATION MODULE ====================
class GovernanceBlockchain:
    """Privacy-first governance blockchain"""
    
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.organizations: Dict[str, Organization] = {}
        self.citizens: Dict[str, Citizen] = {}
        self.government_keys = CryptoUtils.generate_keypair()
        
        # Create genesis block
        self._create_genesis_block()
        
    def _create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0",

        )
        genesis_block.hash = self._calculate_block_hash(genesis_block)
        self.chain.append(genesis_block)
        
    def _calculate_block_hash(self, block, salt=None):
        block_data = {
            "index": block.index,
            "timestamp": block.timestamp,
            "transactions": block.transactions,
            "previous_hash": block.previous_hash,
            "salt": salt,
        }
        block_string = str(block_data).encode()
        return hashlib.sha256(block_string).hexdigest()
        
    def _calculate_transaction_hash(self, tx: Transaction) -> str:
        """Calculate transaction hash"""
        tx_string = json.dumps({
            "tx_id": tx.tx_id,
            "timestamp": tx.timestamp,
            "tx_type": tx.tx_type,
            "data": tx.data
        }, sort_keys=True)
        return CryptoUtils.hash_data(tx_string)
    
    def register_organization(self, name: str, org_type: str) -> Organization:
        """Register a new organization"""
        public_key, private_key = CryptoUtils.generate_keypair()
        org_id = f"org_{len(self.organizations)}_{org_type}"
        
        org = Organization(
            id=org_id,
            name=name,
            type=org_type,
            public_key=public_key,
            private_key=private_key
        )
        
        self.organizations[org_id] = org
        print(f"✅ Registered organization: {name} ({org_type})")
        return org
    
    def register_citizen(self, real_id: str) -> Citizen:
        """Register a new citizen with privacy protection"""
        hashed_id = CryptoUtils.hash_citizen_id(real_id)
        public_key, private_key = CryptoUtils.generate_keypair()
        
        citizen = Citizen(
            real_id=real_id,
            hashed_id=hashed_id,
            public_key=public_key,
            private_key=private_key
        )
        
        self.citizens[hashed_id] = citizen
        print(f"✅ Registered citizen with hashed ID: {hashed_id[:16]}...")
        return citizen
    
    def create_federated_alert(self, org_ids: List[str], alert_data: Dict) -> Transaction:
        """Create a federated alert signed by multiple organizations"""
        tx_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Create transaction data
        tx_data = {
            "alert_type": alert_data.get("type", "general"),
            "severity": alert_data.get("severity", "medium"),
            "message": alert_data["message"],
            "affected_regions": alert_data.get("regions", []),
            "organizations": org_ids,
            "expiry": timestamp + alert_data.get("duration", 86400)  # 24h default
        }
        
        # Create transaction
        transaction = Transaction(
            tx_id=tx_id,
            timestamp=timestamp,
            tx_type="federated_alert",
            data=tx_data,
            signatures={}
        )
        
        # Get signatures from all organizations
        tx_string = json.dumps({
            "tx_id": tx_id,
            "timestamp": timestamp,
            "tx_type": "federated_alert",
            "data": tx_data
        }, sort_keys=True)
        
        for org_id in org_ids:
            if org_id in self.organizations:
                org = self.organizations[org_id]
                signature = CryptoUtils.sign_data(tx_string, org.private_key)
                transaction.signatures[org_id] = signature
        
        transaction.hash = self._calculate_transaction_hash(transaction)
        self.pending_transactions.append(transaction)
        
        print(f"🚨 Created federated alert: {alert_data['message'][:50]}...")
        print(f"   Signed by {len(transaction.signatures)} organizations")
        return transaction
    
    def submit_citizen_feedback(self, citizen_id: str, feedback_data: Dict, anonymous: bool = False) -> Transaction:
        """Submit citizen feedback to the ledger"""
        if citizen_id not in self.citizens:
            raise ValueError("Citizen not registered")
        
        citizen = self.citizens[citizen_id]
        tx_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Prepare feedback data with privacy controls
        tx_data = {
            "feedback_type": feedback_data.get("type", "general"),
            "subject": feedback_data["subject"],
            "content": feedback_data["content"],
            "rating": feedback_data.get("rating"),
            "anonymous": anonymous,
            "citizen_id": "anonymous" if anonymous else citizen.hashed_id,
            "timestamp": timestamp
        }
        
        # Create transaction
        transaction = Transaction(
            tx_id=tx_id,
            timestamp=timestamp,
            tx_type="citizen_feedback",
            data=tx_data,
            signatures={}
        )
        
        # Sign with citizen's key
        tx_string = json.dumps({
            "tx_id": tx_id,
            "timestamp": timestamp,
            "tx_type": "citizen_feedback",
            "data": tx_data
        }, sort_keys=True)
        
        signature = CryptoUtils.sign_data(tx_string, citizen.private_key)
        transaction.signatures[citizen.hashed_id] = signature
        
        transaction.hash = self._calculate_transaction_hash(transaction)
        self.pending_transactions.append(transaction)
        
        feedback_type = "anonymous" if anonymous else "public"
        print(f"💬 Submitted {feedback_type} citizen feedback: {feedback_data['subject']}")
        return transaction
    
    def log_government_action(self, action_data: Dict) -> Transaction:
        """Log government action with accountability"""
        tx_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Create action log data
        tx_data = {
            "action_type": action_data["type"],
            "department": action_data.get("department", "general"),
            "description": action_data["description"],
            "budget_impact": action_data.get("budget_impact", 0),
            "affected_citizens": action_data.get("affected_citizens", []),
            "rationale": action_data.get("rationale", ""),
            "expected_outcome": action_data.get("expected_outcome", ""),
            "timestamp": timestamp,
            "official_id": action_data.get("official_id", "system")
        }
        
        # Create transaction
        transaction = Transaction(
            tx_id=tx_id,
            timestamp=timestamp,
            tx_type="government_action",
            data=tx_data,
            signatures={}
        )
        
        # Sign with government key
        tx_string = json.dumps({
            "tx_id": tx_id,
            "timestamp": timestamp,
            "tx_type": "government_action",
            "data": tx_data
        }, sort_keys=True)
        
        signature = CryptoUtils.sign_data(tx_string, self.government_keys[1])
        transaction.signatures["government"] = signature
        
        transaction.hash = self._calculate_transaction_hash(transaction)
        self.pending_transactions.append(transaction)
        
        print(f"🏛️  Logged government action: {action_data['description'][:50]}...")
        return transaction
    
    def mine_block(self) -> Block:
        """Mine a new block with pending transactions"""
        if not self.pending_transactions:
            print("⚠️  No pending transactions to mine")
            return None
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=self.chain[-1].hash,
            
        )
        
        # Simple proof of work (for simulation)
        while True:
            salt = random.random()
            new_block.hash = self._calculate_block_hash(new_block, salt)
            if new_block.hash.startswith("00"):
                break
        
        # Add block to chain
        self.chain.append(new_block)
        self.pending_transactions.clear()
        
        print(f"⛏️  Mined block #{new_block.index} with {len(new_block.transactions)} transactions")
        print(f"   Block hash: {new_block.hash}")
        return new_block
    
    def verify_transaction_signatures(self, tx: Transaction) -> Dict[str, bool]:
        """Verify all signatures in a transaction"""
        verification_results = {}
        
        tx_string = json.dumps({
            "tx_id": tx.tx_id,
            "timestamp": tx.timestamp,
            "tx_type": tx.tx_type,
            "data": tx.data
        }, sort_keys=True)
        
        for signer_id, signature in tx.signatures.items():
            if signer_id == "government":
                public_key = self.government_keys[0]
            elif signer_id in self.organizations:
                public_key = self.organizations[signer_id].public_key
            elif signer_id in self.citizens:
                public_key = self.citizens[signer_id].public_key
            else:
                verification_results[signer_id] = False
                continue
            
            verification_results[signer_id] = CryptoUtils.verify_signature(
                tx_string, signature, public_key
            )
        
        return verification_results
    
    def verify_blockchain_integrity(self) -> Dict[str, any]:
        """Verify the integrity of the entire blockchain"""
        results = {
            "valid": True,
            "total_blocks": len(self.chain),
            "block_verification": [],
            "transaction_verification": [],
            "hash_chain_valid": True
        }
        
        # Verify each block
        for i, block in enumerate(self.chain):
            block_valid = True
            
            # Verify block hash
            calculated_hash = self._calculate_block_hash(block)
            if calculated_hash != block.hash:
                block_valid = False
                results["valid"] = False
            
            # Verify previous hash chain
            if i > 0 and block.previous_hash != self.chain[i-1].hash:
                block_valid = False
                results["valid"] = False
                results["hash_chain_valid"] = False
            
            results["block_verification"].append({
                "block_index": i,
                "valid": block_valid,
                "hash_valid": calculated_hash == block.hash
            })
            
            # Verify transactions in block
            for tx in block.transactions:
                tx_verification = self.verify_transaction_signatures(tx)
                tx_valid = all(tx_verification.values())
                
                results["transaction_verification"].append({
                    "tx_id": tx.tx_id,
                    "tx_type": tx.tx_type,
                    "valid": tx_valid,
                    "signature_verification": tx_verification
                })
                
                if not tx_valid:
                    results["valid"] = False
        
        return results
# ============================================================================
# SIMULATION AND REPORTING
# ============================================================================
class GovernanceSimulation:
    """Complete governance system simulation"""
    
    def __init__(self):
        self.blockchain = GovernanceBlockchain()
        self.simulation_data = {
            "alerts_created": 0,
            "feedback_submitted": 0,
            "actions_logged": 0,
            "blocks_mined": 0
        }
    
    def run_complete_simulation(self):
        """Run a complete simulation of the governance system"""
        print(" Starting Privacy-First Governance Blockchain Simulation")
        print("=" * 60)
        
        # Phase 1: Setup
        print("\n PHASE 1: SYSTEM SETUP")
        self._setup_organizations()
        self._setup_citizens()
        
        # Phase 2: Federated Alerts
        print("\n PHASE 2: FEDERATED ALERTS")
        self._simulate_federated_alerts()
        
        # Phase 3: Citizen Feedback
        print("\n PHASE 3: CITIZEN FEEDBACK")
        self._simulate_citizen_feedback()
        
        # Phase 4: Government Actions
        print("\n PHASE 4: GOVERNMENT ACTIONS")
        self._simulate_government_actions()
        
        # Phase 5: Mining
        print("\n  PHASE 5: BLOCKCHAIN MINING")
        self._mine_transactions()
        
        # Phase 6: Verification and Reports
        print("\n PHASE 6: VERIFICATION & REPORTS")
        self._run_verification()
        self._generate_reports()
        
        print("\n SIMULATION COMPLETE!")
        print("=" * 60)
    
    def _setup_organizations(self):
        """Setup federated organizations"""
        orgs = [
            ("Climate Research Institute", "climate"),
            ("National Weather Service", "climate"),
            ("Emergency Management Agency", "emergency"),
            ("Public Health Department", "health"),
            ("Environmental Protection Agency", "climate")
        ]
        
        for name, org_type in orgs:
            self.blockchain.register_organization(name, org_type)
    
    def _setup_citizens(self):
        """Setup citizens"""
        citizens = [
            "citizen_alice_001",
            "citizen_bob_002", 
            "citizen_carol_003",
            "citizen_david_004",
            "citizen_eve_005"
        ]
        
        for citizen_id in citizens:
            self.blockchain.register_citizen(citizen_id)
    
    def _simulate_federated_alerts(self):
        """Simulate federated alert creation"""
        alerts = [
            {
                "orgs": ["org_0_climate", "org_1_climate"],
                "data": {
                    "type": "extreme_weather",
                    "severity": "high",
                    "message": "Severe thunderstorm warning for metropolitan area",
                    "regions": ["downtown", "suburbs"],
                    "duration": 7200
                }
            },
            {
                "orgs": ["org_2_emergency", "org_3_health"],
                "data": {
                    "type": "public_health",
                    "severity": "medium",
                    "message": "Air quality alert - sensitive groups should limit outdoor activities",
                    "regions": ["industrial_district"],
                    "duration": 14400
                }
            },
            {
                "orgs": ["org_0_climate", "org_4_climate", "org_2_emergency"],
                "data": {
                    "type": "flood_warning",
                    "severity": "critical",
                    "message": "Flash flood warning - evacuate low-lying areas immediately",
                    "regions": ["riverside", "lowlands"],
                    "duration": 10800
                }
            }
        ]
        
        for alert in alerts:
            self.blockchain.create_federated_alert(alert["orgs"], alert["data"])
            self.simulation_data["alerts_created"] += 1
    
    def _simulate_citizen_feedback(self):
        """Simulate citizen feedback submission"""
        citizens = list(self.blockchain.citizens.keys())
        
        feedback_examples = [
            {
                "type": "service_quality",
                "subject": "Emergency Response Time",
                "content": "The emergency response team arrived within 5 minutes. Excellent service!",
                "rating": 5,
                "anonymous": False
            },
            {
                "type": "infrastructure",
                "subject": "Road Maintenance Issue",
                "content": "Pothole on Main Street needs urgent repair. Creating traffic hazard.",
                "rating": 2,
                "anonymous": True
            },
            {
                "type": "public_safety",
                "subject": "Flood Warning System",
                "content": "The new flood warning system worked perfectly. Got alerts 2 hours before flooding.",
                "rating": 5,
                "anonymous": False
            },
            {
                "type": "complaint",
                "subject": "Noise Pollution",
                "content": "Construction noise starting too early in residential areas. Please regulate hours.",
                "rating": 2,
                "anonymous": True
            },
            {
                "type": "suggestion",
                "subject": "Public Transport",
                "content": "Adding more electric buses would reduce emissions and improve air quality.",
                "rating": 4,
                "anonymous": False
            }
        ]
        
        for i, feedback in enumerate(feedback_examples):
            citizen_id = citizens[i % len(citizens)]
            self.blockchain.submit_citizen_feedback(citizen_id, feedback, feedback["anonymous"])
            self.simulation_data["feedback_submitted"] += 1
    
    def _simulate_government_actions(self):
        """Simulate government action logging"""
        actions = [
            {
                "type": "emergency_response",
                "department": "Emergency Management",
                "description": "Deployed emergency shelters in flood-affected areas",
                "budget_impact": 250000,
                "affected_citizens": ["riverside", "lowlands"],
                "rationale": "Response to critical flood warning alert",
                "expected_outcome": "Safe shelter for 500+ displaced residents"
            },
            {
                "type": "infrastructure",
                "department": "Public Works",
                "description": "Initiated road repair program based on citizen feedback",
                "budget_impact": 75000,
                "affected_citizens": ["downtown"],
                "rationale": "Multiple citizen complaints about road conditions",
                "expected_outcome": "Improved road safety and reduced vehicle damage"
            },
            {
                "type": "policy_change",
                "department": "Environmental",
                "description": "Updated construction noise regulations",
                "budget_impact": 0,
                "affected_citizens": ["all_residential"],
                "rationale": "Citizen feedback regarding early morning construction noise",
                "expected_outcome": "Reduced noise complaints and improved quality of life"
            },
            {
                "type": "investment",
                "department": "Transportation",
                "description": "Approved budget for 20 new electric buses",
                "budget_impact": 2500000,
                "affected_citizens": ["citywide"],
                "rationale": "Citizen suggestion to reduce emissions",
                "expected_outcome": "30% reduction in public transport emissions"
            }
        ]
        
        for action in actions:
            self.blockchain.log_government_action(action)
            self.simulation_data["actions_logged"] += 1
    
    def _mine_transactions(self):
        """Mine all pending transactions"""
        while self.blockchain.pending_transactions:
            block = self.blockchain.mine_block()
            if block:
                self.simulation_data["blocks_mined"] += 1
    
    def _run_verification(self):
        """Run complete system verification"""
        print("\n🔍 Running comprehensive verification...")
        
        # Verify blockchain integrity
        integrity_results = self.blockchain.verify_blockchain_integrity()
        
        if integrity_results["valid"]:
            print("✅ Blockchain integrity: VALID")
        else:
            print("❌ Blockchain integrity: INVALID")
        
        print(f"   Total blocks verified: {integrity_results['total_blocks']}")
        print(f"   Hash chain valid: {integrity_results['hash_chain_valid']}")
        
        # Verify individual transactions
        valid_transactions = sum(1 for tx in integrity_results["transaction_verification"] if tx["valid"])
        total_transactions = len(integrity_results["transaction_verification"])
        print(f"   Valid transactions: {valid_transactions}/{total_transactions}")
        
        # Test tamper detection
        print("\n🔒 Testing tamper detection...")
        self._test_tamper_detection()
    
    def _test_tamper_detection(self):
        """Test the system's ability to detect tampering"""
        if len(self.blockchain.chain) > 1:
            # Simulate tampering with a transaction
            original_content = self.blockchain.chain[1].transactions[0].data["content"] if self.blockchain.chain[1].transactions else None
            
            if original_content:
                print(f"   Original content: {original_content[:30]}...")
                
                # Tamper with the data
                self.blockchain.chain[1].transactions[0].data["content"] = "TAMPERED CONTENT"
                
                # Verify integrity (should detect tampering)
                results = self.blockchain.verify_blockchain_integrity()
                
                if not results["valid"]:
                    print("✅ Tamper detection: SUCCESS (tampering detected)")
                else:
                    print("❌ Tamper detection: FAILED (tampering not detected)")
                
                # Restore original content
                self.blockchain.chain[1].transactions[0].data["content"] = original_content
    
    def _generate_reports(self):
        """Generate comprehensive system reports"""
        print("\n📊 SYSTEM REPORTS")
        print("-" * 30)
        
        # Simulation Statistics
        print("📈 Simulation Statistics:")
        for key, value in self.simulation_data.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        # Privacy Report
        print("\n🔒 Privacy Protection Report:")
        anonymous_feedback = 0
        public_feedback = 0
        
        for block in self.blockchain.chain:
            for tx in block.transactions:
                if tx.tx_type == "citizen_feedback":
                    if tx.data.get("anonymous", False):
                        anonymous_feedback += 1
                    else:
                        public_feedback += 1
        
        print(f"   Anonymous feedback: {anonymous_feedback}")
        print(f"   Public feedback: {public_feedback}")
        print(f"   Citizens registered: {len(self.blockchain.citizens)}")
        print("   ✅ All citizen IDs are hashed for privacy")
        
        # Security Report
        print("\n🛡️  Security Report:")
        print(f"   Organizations registered: {len(self.blockchain.organizations)}")
        print("   ✅ All transactions cryptographically signed")
        print("   ✅ Multi-signature federated alerts implemented")
        print("   ✅ Government actions have accountability signatures")
        
        # Blockchain Statistics
        print("\n⛓️  Blockchain Statistics:")
        total_transactions = sum(len(block.transactions) for block in self.blockchain.chain)
        print(f"   Total blocks: {len(self.blockchain.chain)}")
        print(f"   Total transactions: {total_transactions}")
        print(f"   Chain size: {len(self.blockchain.chain)} blocks")
        
        # Transaction Type Breakdown
        tx_types = {}
        for block in self.blockchain.chain:
            for tx in block.transactions:
                tx_types[tx.tx_type] = tx_types.get(tx.tx_type, 0) + 1
        
        print("\n📋 Transaction Types:")
        for tx_type, count in tx_types.items():
            print(f"   {tx_type.replace('_', ' ').title()}: {count}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    try:
        # Run the complete simulation
        simulation = GovernanceSimulation()
        simulation.run_complete_simulation()
        
        print("\n All systems operational!")
        print("The privacy-first governance blockchain is ready for production.")
        
    except Exception as e:
        print(f" Simulation error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Install required packages
    try:
        from cryptography.hazmat.primitives import hashes
    except ImportError:
        print("Installing required cryptography package...")
        import subprocess
        subprocess.check_call(["pip", "install", "cryptography"])
        from cryptography.hazmat.primitives import hashes
    
main()
# ==================== RAG SYSTEM MODULE ====================
class ClimateKnowledgeBase:
    """RAG-based climate knowledge system"""

    def __init__(self):
        self.documents = []
        self.embeddings = {}
        self.load_climate_knowledge()
        logger.info("Initialized Climate Knowledge Base with RAG")

    def load_climate_knowledge(self):
        """Load climate disaster knowledge base"""
        knowledge_base = [
            {
                "id": "flood_001",
                "content": "Floods are caused by heavy rainfall, dam failures, or coastal storm surges. Early warning signs include rapid water level rise, heavy continuous rainfall for 24+ hours, and upstream dam alerts.",
                "disaster_type": "flood",
                "region": "general"
            },
            {
                "id": "heatwave_001",
                "content": "Heatwaves are prolonged periods of excessively hot weather. In India, temperatures above 40°C for 3+ days constitute a heatwave. Vulnerable populations include elderly, children, and outdoor workers.",
                "disaster_type": "heatwave",
                "region": "india"
            },
            {
                "id": "cyclone_001",
                "content": "Cyclones in the Bay of Bengal typically occur between April-June and October-December. Warning signs include sudden drop in barometric pressure, increasing wind speeds, and heavy cloud formation.",
                "disaster_type": "cyclone",
                "region": "coastal_india"
            },
            {
                "id": "drought_001",
                "content": "Droughts occur when rainfall is significantly below normal for extended periods. Agricultural drought affects crop production, while meteorological drought refers to precipitation deficits.",
                "disaster_type": "drought",
                "region": "general"
            }
        ]

        self.documents = knowledge_base
        # In real implementation, you'd use sentence transformers for embeddings
        for doc in knowledge_base:
            # Simulate embedding with simple hash for demo
            self.embeddings[doc["id"]] = hash(doc["content"]) % 1000

    def query_knowledge(self, query: str, disaster_type: str = None) -> List[Dict]:
        """Query knowledge base using RAG"""
        # Simple similarity search simulation
        # In real implementation, use FAISS or similar vector DB
        relevant_docs = []

        for doc in self.documents:
            if disaster_type and doc["disaster_type"] != disaster_type:
                continue

            # Simple keyword matching for demo
            if any(word.lower() in doc["content"].lower() for word in query.split()):
                relevant_docs.append(doc)

        return relevant_docs[:3]  # Return top 3 relevant documents
    
# ==================== CHAT BOT ====================

class VoiceProcessor:
    def __init__(self):
        self.engine = pyttsx3.init()
        
        # Windows-specific optimizations
        if platform.system() == "Windows":
            # Try to use SAPI5 voice engine for better Windows compatibility
            try:
                voices = self.engine.getProperty('voices')
                if voices:
                    # Use the first available voice
                    self.engine.setProperty('voice', voices[0].id)
            except:
                pass
    
    def text_to_speech(self, text: str, language: str = "en", output_file: str = None, speed: int = 150):
        try:
            # Set speech rate
            self.engine.setProperty('rate', speed)
            
            # Set language-specific voice if available
            if language != "en":
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if language in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # Ensure the directory exists
            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Save to file
                self.engine.save_to_file(text, output_file)
                self.engine.runAndWait()
                
                # Verify file was created and has content
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    print(f"Audio file created successfully: {output_file}")
                    print(f"File size: {os.path.getsize(output_file)} bytes")
                    return True
                else:
                    print(f"Audio file creation failed or file is empty: {output_file}")
                    return False
            else:
                # Just speak without saving
                self.engine.say(text)
                self.engine.runAndWait()
                return True
                
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return False

        
class DisasterResponseBot:
    def __init__(self):
        # API Keys from environment variables
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.weather_api_key = os.getenv("OPENWEATHER_API_KEY")
        
        # Validate API keys
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables!")
            
        if not self.weather_api_key:
            raise ValueError("WEATHER_API_KEY not found in environment variables!")
        
        # Initialize Gemini
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')

         # Initialize voice processor
        self.voice_processor = VoiceProcessor()
        
        # Initialize Granite LLM (via HuggingFace) - Optional
        self.granite_llm = None
        try:
            self.granite_llm = pipeline(
                "text-generation", 
                model="ibm-granite/granite-3b-code-instruct", 
                trust_remote_code=True,
                device_map="auto"
            )
        except Exception as e:
            print(f"Warning: Granite LLM not available: {e}")
        
        # Language support
        self.languages = {
            "English": {"code": "en", "flag": "🇺🇸"},
            "हिंदी": {"code": "hi", "flag": "🇮🇳"},  
            "ಕನ್ನಡ": {"code": "kn", "flag": "🇮🇳"},
            "తెలుగు": {"code": "te", "flag": "🇮🇳"},
            "தமிழ்": {"code": "ta", "flag": "🇮🇳"},
            "বাংলা": {"code": "bn", "flag": "🇧🇩"}
        }
        
        # Disaster knowledge base
        self.knowledge_base = {
            "flood": {
                "en": "🌊 **FLOOD SAFETY:**\n• Move to higher ground immediately\n• Avoid walking/driving through flooded areas\n• Stay away from electrical equipment if you're wet\n• Listen to emergency broadcasts\n• Have emergency supplies ready",
                "hi": "🌊 **बाढ़ की सुरक्षा:**\n• तुरंत ऊंची जगह पर जाएं\n• बाढ़ के पानी में चलने/गाड़ी चलाने से बचें\n• गीले होने पर बिजली के उपकरणों से दूर रहें\n• आपातकालीन प्रसारण सुनें\n• आपातकालीन सामान तैयार रखें",
                "kn": "🌊 **ಪ್ರವಾಹ ಸುರಕ್ಷತೆ:**\n• ತಕ್ಷಣ ಎತ್ತರದ ಸ್ಥಳಕ್ಕೆ ಹೋಗಿ\n• ಪ್ರವಾಹದ ನೀರಿನಲ್ಲಿ ನಡೆಯುವುದು/ವಾಹನ ಚಲಾಯಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ\n• ಒದ್ದೆಯಾಗಿದ್ದರೆ ವಿದ್ಯುತ್ ಉಪಕರಣಗಳಿಂದ ದೂರವಿರಿ\n• ತುರ್ತು ಪ್ರಸಾರವನ್ನು ಕೇಳಿ\n• ತುರ್ತು ಸಾಮಗ್ರಿಗಳನ್ನು ಸಿದ್ಧಪಡಿಸಿ"
            },
            "earthquake": {
                "en": "🏠 **EARTHQUAKE SAFETY:**\n• Drop, Cover, Hold On!\n• Get under a sturdy table\n• Stay away from windows and heavy objects\n• If outdoors, move away from buildings\n• After shaking stops, evacuate if building is damaged",
                "hi": "🏠 **भूकंप की सुरक्षा:**\n• झुकें, छुपें, पकड़ें!\n• मजबूत मेज के नीचे जाएं\n• खिड़कियों और भारी वस्तुओं से दूर रहें\n• बाहर हों तो इमारतों से दूर जाएं\n• हिलना बंद होने पर क्षतिग्रस्त इमारत से बाहर निकलें",
                "kn": "🏠 **ಭೂಕಂಪ ಸುರಕ್ಷತೆ:**\n• ಕೆಳಗೆ ಬಿದ್ದು, ಮರೆಯಾಗಿ, ಹಿಡಿದುಕೊಳ್ಳಿ!\n• ದೃಢವಾದ ಮೇಜಿನ ಕೆಳಗೆ ಹೋಗಿ\n• ಕಿಟಕಿಗಳು ಮತ್ತು ಭಾರವಾದ ವಸ್ತುಗಳಿಂದ ದೂರವಿರಿ\n• ಹೊರಗಿದ್ದರೆ ಕಟ್ಟಡಗಳಿಂದ ದೂರ ಹೋಗಿ\n• ಅಲುಗಾಟ ನಿಂತ ನಂತರ ಹಾನಿಗೊಳಗಾದ ಕಟ್ಟಡದಿಂದ ಹೊರಬನ್ನಿ"
            },
            "heatwave": {
                "en": "☀️ **HEATWAVE SAFETY:**\n• Stay indoors during peak hours (10am-4pm)\n• Drink plenty of water regularly\n• Wear light-colored, loose clothing\n• Use fans, AC, or cool showers\n• Check on elderly neighbors",
                "hi": "☀️ **लू की सुरक्षा:**\n• चरम घंटों (सुबह 10-शाम 4) के दौरान घर के अंदर रहें\n• नियमित रूप से भरपूर पानी पिएं\n• हल्के रंग के ढीले कपड़े पहनें\n• पंखे, AC, या ठंडे पानी से नहाएं\n• बुजुर्ग पड़ोसियों की जांच करें",
                "kn": "☀️ **ಶಾಖದ ಅಲೆ ಸುರಕ್ಷತೆ:**\n• ಗರಿಷ್ಠ ಸಮಯದಲ್ಲಿ (ಬೆಳಿಗ್ಗೆ 10-ಸಂಜೆ 4) ಮನೆಯೊಳಗೆ ಇರಿ\n• ನಿಯಮಿತವಾಗಿ ಸಾಕಷ್ಟು ನೀರು ಕುಡಿಯಿರಿ\n• ತಿಳಿ ಬಣ್ಣದ, ಸಡಿಲವಾದ ಬಟ್ಟೆಗಳನ್ನು ಧರಿಸಿ\n• ಫ್ಯಾನ್, AC, ಅಥವಾ ತಣ್ಣನೆಯ ಸ್ನಾನ ಮಾಡಿ\n• ವಯಸ್ಸಾದ ನೆರೆಹೊರೆಯವರನ್ನು ಪರಿಶೀಲಿಸಿ"
            },
            "cyclone": {
                "en": "🌪️ **CYCLONE SAFETY:**\n• Stay indoors and away from windows\n• Store water and non-perishable food\n• Charge all electronic devices\n• Keep battery radio for updates\n• Secure outdoor items",
                "hi": "🌪️ **चक्रवात की सुरक्षा:**\n• घर के अंदर रहें और खिड़कियों से दूर रहें\n• पानी और सूखा खाना स्टोर करें\n• सभी उपकरण चार्ज करें\n• अपडेट के लिए बैटरी रेडियो रखें\n• बाहरी वस्तुओं को सुरक्षित करें",
                "kn": "🌪️ **ಚಂಡಮಾರುತ ಸುರಕ್ಷತೆ:**\n• ಮನೆಯೊಳಗೆ ಇರಿ ಮತ್ತು ಕಿಟಕಿಗಳಿಂದ ದೂರವಿರಿ\n• ನೀರು ಮತ್ತು ಕೆಡದ ಆಹಾರವನ್ನು ಶೇಖರಿಸಿ\n• ಎಲ್ಲಾ ಸಾಧನಗಳನ್ನು ಚಾರ್ಜ್ ಮಾಡಿ\n• ಅಪ್‌ಡೇಟ್‌ಗಳಿಗಾಗಿ ಬ್ಯಾಟರಿ ರೇಡಿಯೋ ಇರಿಸಿ\n• ಹೊರಗಿನ ವಸ್ತುಗಳನ್ನು ಸುರಕ್ಷಿತಗೊಳಿಸಿ"
            }
        }
        
        # Emergency contacts
        self.emergency_contacts = {
            "en": """🚨 **EMERGENCY NUMBERS:**
            
**India Emergency Services:**
• **Police:** 100 📞
• **Fire Brigade:** 101 🚒
• **Ambulance:** 108 🚑
• **Disaster Management:** 1070 🌪️
• **Women Helpline:** 1091 👩
• **Child Helpline:** 1098 👶
• **Tourist Emergency:** 1363 🧳

**Additional Resources:**
• **Blood Bank:** 104
• **Poison Control:** 1066""",
            
            "hi": """🚨 **आपातकालीन नंबर:**
            
**भारत आपातकालीन सेवाएं:**
• **पुलिस:** 100 📞
• **दमकल:** 101 🚒
• **एम्बुलेंस:** 108 🚑
• **आपदा प्रबंधन:** 1070 🌪️
• **महिला हेल्पलाइन:** 1091 👩
• **बाल हेल्पलाइन:** 1098 👶
• **पर्यटक आपातकाल:** 1363 🧳

**अतिरिक्त संसाधन:**
• **ब्लड बैंक:** 104
• **जहर नियंत्रण:** 1066""",
            
            "kn": """🚨 **ತುರ್ತು ಸಂಖ್ಯೆಗಳು:**
            
**ಭಾರತದ ತುರ್ತು ಸೇವೆಗಳು:**
• **ಪೊಲೀಸ್:** 100 📞
• **ಅಗ್ನಿಶಾಮಕ:** 101 🚒
• **ಆಂಬ್ಯುಲೆನ್ಸ್:** 108 🚑
• **ವಿಪತ್ತು ನಿರ್ವಹಣೆ:** 1070 🌪️
• **ಮಹಿಳಾ ಸಹಾಯವಾಣಿ:** 1091 👩
• **ಮಕ್ಕಳ ಸಹಾಯವಾಣಿ:** 1098 👶
• **ಪ್ರವಾಸಿ ತುರ್ತು:** 1363 🧳

**ಹೆಚ್ಚುವರಿ ಸಂಪನ್ಮೂಲಗಳು:**
• **ರಕ್ತ ಬ್ಯಾಂಕ್:** 104
• **ವಿಷ ನಿಯಂತ್ರಣ:** 1066"""
        }
        
        # Initialize feedback storage
        self.feedback_file = "disaster_bot_feedback.csv"
        self.init_feedback_storage()
    
    def init_feedback_storage(self):
        """Initialize CSV file for feedback storage"""
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'language', 'location', 'safety_status', 'govt_rating', 'feedback'])
    
    def translate_text(self, text: str, target_lang: str) -> str:
        """Simple translation using LibreTranslate"""
        if target_lang == "en":
            return text
        
        try:
            url = "https://libretranslate.de/translate"
            data = {
                "q": text,
                "source": "en",
                "target": target_lang,
                "format": "text"
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                return response.json()["translatedText"]
        except:
            pass
        
        return text
    
    def get_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather information"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                temp = data['main']['temp']
                description = data['weather'][0]['description'].title()
                humidity = data['main']['humidity']
                feels_like = data['main']['feels_like']
                wind_speed = data['wind']['speed']
                
                weather_info = f"""🌤️ **Current Weather in {location}:**
                
**Temperature:** {temp}°C (Feels like {feels_like}°C)
**Condition:** {description}
**Humidity:** {humidity}%
**Wind Speed:** {wind_speed} m/s
                
*Last updated: {datetime.now().strftime('%I:%M %p')}*"""
                
                return {"success": True, "weather_info": weather_info}
        except Exception as e:
            return {
                "success": False, 
                "weather_info": f"❌ Unable to fetch weather for {location}. Please check the location name."
            }
    
    def get_disaster_advice(self, disaster_type: str, language: str) -> Optional[str]:
        """Get disaster-specific advice from knowledge base"""
        disaster_type = disaster_type.lower()
        for key in self.knowledge_base:
            if key in disaster_type:
                return self.knowledge_base[key].get(language, self.knowledge_base[key]["en"])
        return None
    
    def use_granite_llm(self, prompt: str) -> Optional[str]:
        """Use Granite LLM for technical/coding questions"""
        if not self.granite_llm:
            return None
        
        try:
            response = self.granite_llm(prompt, max_length=300, do_sample=True, temperature=0.7, pad_token_id=50256)
            return response[0]['generated_text'][len(prompt):].strip()
        except:
            return None
    
    def use_gemini(self, prompt: str) -> str:
        """Use Gemini for general questions"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I'm having trouble connecting to my knowledge base: {str(e)}"
    
    def get_ai_response(self, question: str, language: str) -> str:
        """Get AI response using RAG-like approach"""
        # Check knowledge base first
        disaster_advice = self.get_disaster_advice(question, language)
        if disaster_advice:
            return disaster_advice
        
        # For technical/coding questions, try Granite LLM
        if any(word in question.lower() for word in ['code', 'programming', 'technical', 'software', 'python', 'javascript']):
            granite_response = self.use_granite_llm(question)
            if granite_response:
                return f"🔧 **Technical Response:**\n\n{granite_response}"
        
        # Use Gemini for general questions
        prompt = f"""You are a helpful disaster response assistant. Answer this question briefly and helpfully: {question}
        
        Focus on:
        - Safety and emergency information
        - Practical, actionable advice
        - Keep response under 250 words
        - Use emojis where appropriate
        - Be supportive and reassuring
        """
        
        response = self.use_gemini(prompt)
        
        # Translate if needed
        if language != "en":
            response = self.translate_text(response, language)
        
        return response
    
    def save_feedback(self, safety_status: str, govt_rating: str, feedback: str, language: str, location: str) -> Dict[str, Any]:
        """Save user feedback to CSV"""
        try:
            with open(self.feedback_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().isoformat(),
                    language,
                    location,
                    safety_status,
                    govt_rating,
                    feedback
                ])
            return {"success": True, "message": "Feedback saved successfully"}
        except Exception as e:
            return {"success": False, "message": f"Error saving feedback: {str(e)}"}
    
    def get_emergency_contacts(self, language: str) -> str:
        """Get emergency contacts in specified language"""
        return self.emergency_contacts.get(language, self.emergency_contacts["en"])
    
    def get_disaster_guide(self, disaster_type: str, language: str) -> Optional[str]:
        """Get disaster guide for specific disaster type"""
        return self.get_disaster_advice(disaster_type, language)
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback summary for analytics"""
        try:
            df = pd.read_csv(self.feedback_file)
            if df.empty:
                return {"success": False, "message": "No feedback data available"}
            
            safety_counts = df['safety_status'].value_counts().to_dict()
            avg_rating = df['govt_rating'].astype(float).mean()
            recent_feedback = df.tail(10).to_dict('records')
            
            return {
                "success": True,
                "data": {
                    "safety_distribution": safety_counts,
                    "average_rating": round(avg_rating, 1),
                    "recent_feedback": recent_feedback,
                    "total_responses": len(df)
                }
            }
        except FileNotFoundError:
            return {"success": False, "message": "No feedback data available"}
        except Exception as e:
            return {"success": False, "message": f"Error reading feedback: {str(e)}"}

# Global bot instance
disaster_bot = None

def get_bot():
    """Get or create bot instance"""
    global disaster_bot
    if disaster_bot is None:
        disaster_bot = DisasterResponseBot()
    return disaster_bot


# ==================== Resilience Score ====================

class GeminiDisasterResilienceAnalyzer:
    def __init__(self):
        self.analysis_history = []
        self.categories = {
            'infrastructure': 'Infrastructure & Built Environment',
            'economic': 'Economic Stability & Diversity', 
            'social': 'Social Cohesion & Human Capital',
            'environmental': 'Environmental & Climate Resilience',
            'governance': 'Governance & Institutional Capacity',
            'emergency': 'Emergency Preparedness & Response'
        }
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def configure_gemini(self, api_key: str):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            return True
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {str(e)}")
            return False
    
    async def get_comprehensive_resilience_analysis(self, location: str, api_key: str) -> Dict:
        """Use Gemini to fetch all relevant data and calculate resilience scores."""
        
        if not self.configure_gemini(api_key):
            raise HTTPException(status_code=400, detail="Invalid Gemini API key")
        
        prompt = f"""
        You are a disaster resilience expert analyzing {location}. Please provide a comprehensive analysis by researching and gathering the following information:

        **DATA TO RESEARCH AND ANALYZE:**

        1. **INFRASTRUCTURE & BUILT ENVIRONMENT (25% weight):**
        - Hospital density and healthcare capacity
        - Transportation network quality (roads, public transit)
        - Building codes and construction standards
        - Utilities infrastructure (power, water, communications)
        - Critical infrastructure vulnerability

        2. **ECONOMIC STABILITY & DIVERSITY (20% weight):**
        - GDP per capita and economic indicators
        - Employment rates and job diversity
        - Income inequality levels
        - Economic recovery capacity after disasters
        - Financial resources for emergency response

        3. **SOCIAL COHESION & HUMAN CAPITAL (15% weight):**
        - Education levels and literacy rates
        - Population demographics and vulnerability
        - Community organization and social networks
        - Crime rates and public safety
        - Cultural resilience and adaptation capacity

        4. **ENVIRONMENTAL & CLIMATE RESILIENCE (15% weight):**
        - Natural disaster risk profile (floods, earthquakes, storms, etc.)
        - Climate change vulnerability and adaptation
        - Environmental degradation levels
        - Green infrastructure and natural barriers
        - Air and water quality indicators

        5. **GOVERNANCE & INSTITUTIONAL CAPACITY (10% weight):**
        - Government effectiveness and corruption levels
        - Policy framework for disaster management
        - Inter-agency coordination capabilities
        - Public trust in institutions
        - Regulatory enforcement capacity

        6. **EMERGENCY PREPAREDNESS & RESPONSE (15% weight):**
        - Emergency response time and capabilities
        - Early warning systems effectiveness
        - Disaster preparedness planning
        - Community training and awareness programs
        - Recovery and reconstruction capacity

        **OUTPUT REQUIRED:**
        Please provide your analysis in the following JSON format:

        {{
            "location_info": {{
                "full_name": "Complete location name",
                "country": "Country name",
                "population": "Current population estimate",
                "area_km2": "Area in square kilometers",
                "coordinates": {{"lat": latitude, "lon": longitude}},
                "key_characteristics": ["characteristic1", "characteristic2", "characteristic3"]
            }},
            "category_scores": {{
                "infrastructure": score_0_to_100,
                "economic": score_0_to_100,
                "social": score_0_to_100,
                "environmental": score_0_to_100,
                "governance": score_0_to_100,
                "emergency": score_0_to_100
            }},
            "overall_resilience_score": overall_score_0_to_100,
            "risk_assessment": {{
                "primary_threats": ["threat1", "threat2", "threat3"],
                "risk_level": "Low/Medium/High/Critical",
                "most_vulnerable_areas": ["area1", "area2"],
                "climate_risks": ["risk1", "risk2"]
            }},
            "strengths": ["strength1", "strength2", "strength3"],
            "vulnerabilities": ["vulnerability1", "vulnerability2", "vulnerability3"],
            "recommendations": {{
                "immediate_actions": ["action1", "action2"],
                "medium_term_improvements": ["improvement1", "improvement2"],
                "long_term_strategic_goals": ["goal1", "goal2"]
            }},
            "data_sources_considered": ["source1", "source2", "source3"],
            "confidence_level": "High/Medium/Low",
            "last_updated": "{datetime.now().isoformat()}"
        }}

        Please ensure all scores are realistic and based on actual conditions. Provide specific, actionable insights rather than generic recommendations.
        """
        
        try:
            # Run Gemini API call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor, 
                lambda: self.model.generate_content(prompt)
            )
            response_text = response.text
            
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end != 0:
                    json_text = response_text[json_start:json_end]
                else:
                    raise ValueError("No JSON structure found in response")
            
            # Parse the JSON response
            analysis_data = json.loads(json_text)
            
            # Store in history
            analysis_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            analysis_data['location_query'] = location
            self.analysis_history.append(analysis_data)
            
            return analysis_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to parse analysis response: {str(e)}")
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    def create_resilience_dashboard(self, analysis_data: Dict) -> Dict:
        """Create comprehensive visualizations for the analysis."""
        
        try:
            # 1. Create Overall Score Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=analysis_data['overall_resilience_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Overall Resilience Score<br>{analysis_data['location_info']['full_name']}"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "navy"},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},
                        {'range': [30, 60], 'color': "orange"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=400, font={'size': 16})
            
            # 2. Create Category Radar Chart
            categories = list(self.categories.keys())
            scores = [analysis_data['category_scores'][cat] for cat in categories]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=scores,
                theta=[self.categories[cat] for cat in categories],
                fill='toself',
                name='Resilience Scores',
                line=dict(color='blue', width=3),
                fillcolor='rgba(0,100,255,0.2)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickmode='linear',
                        tick0=0,
                        dtick=20
                    )),
                showlegend=True,
                title=f"Category Breakdown - {analysis_data['location_info']['full_name']}",
                height=500,
                font={'size': 12}
            )
            
            # 3. Create Category Bar Chart
            fig_bar = px.bar(
                x=list(self.categories.values()),
                y=scores,
                color=scores,
                color_continuous_scale='RdYlGn',
                title=f"Resilience Scores by Category - {analysis_data['location_info']['full_name']}",
                labels={'x': 'Categories', 'y': 'Score (0-100)'}
            )
            fig_bar.update_layout(height=400, xaxis_tickangle=-45)
            
            # 4. Create Comparison Chart if multiple analyses exist
            if len(self.analysis_history) > 1:
                comparison_data = []
                for analysis in self.analysis_history[-5:]:  # Last 5 analyses
                    comparison_data.append({
                        'Location': analysis['location_info']['full_name'][:20],
                        'Overall Score': analysis['overall_resilience_score'],
                        'Infrastructure': analysis['category_scores']['infrastructure'],
                        'Economic': analysis['category_scores']['economic'],
                        'Social': analysis['category_scores']['social'],
                        'Environmental': analysis['category_scores']['environmental'],
                        'Governance': analysis['category_scores']['governance'],
                        'Emergency': analysis['category_scores']['emergency']
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                
                fig_comparison = px.bar(
                    df_comparison,
                    x='Location',
                    y='Overall Score',
                    color='Overall Score',
                    color_continuous_scale='RdYlGn',
                    title="Comparison of Recent Analyses",
                    height=400
                )
                fig_comparison.update_layout(xaxis_tickangle=-45)
            else:
                fig_comparison = go.Figure()
                fig_comparison.add_annotation(
                    text="Analyze more locations to see comparisons",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font_size=16
                )
                fig_comparison.update_layout(height=400, title="Location Comparison")
            
            return {
                "gauge_chart": json.loads(fig_gauge.to_json()),
                "radar_chart": json.loads(fig_radar.to_json()),
                "bar_chart": json.loads(fig_bar.to_json()),
                "comparison_chart": json.loads(fig_comparison.to_json())
            }
            
        except Exception as e:
            logger.error(f"Dashboard creation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create dashboard: {str(e)}")
    
    def create_detailed_report(self, analysis_data: Dict) -> str:
        """Create a detailed markdown report."""
        
        location_info = analysis_data['location_info']
        risk_info = analysis_data['risk_assessment']
        recommendations = analysis_data['recommendations']
        
        report = f"""
# 🏘️ Disaster Resilience Analysis Report
## {location_info['full_name']}

---

### 📍 **Location Overview**
- **Country:** {location_info['country']}
- **Population:** {location_info['population']}
- **Area:** {location_info['area_km2']} km²
- **Key Characteristics:** {', '.join(location_info['key_characteristics'])}

---

### 🎯 **Overall Resilience Score: {analysis_data['overall_resilience_score']}/100**

**Risk Level:** {risk_info['risk_level']}

---

### 📊 **Category Scores**

"""
        
        # Add category scores
        for cat_key, cat_name in self.categories.items():
            score = analysis_data['category_scores'][cat_key]
            if score >= 80:
                emoji = "🟢"
            elif score >= 60:
                emoji = "🟡"
            elif score >= 40:  
                emoji = "🟠"
            else:
                emoji = "🔴"
                
            report += f"- **{cat_name}:** {emoji} {score}/100\n"
        
        report += f"""

---

### 🚨 **Risk Assessment**

**Primary Threats:**
{chr(10).join(['• ' + threat for threat in risk_info['primary_threats']])}

**Most Vulnerable Areas:**
{chr(10).join(['• ' + area for area in risk_info['most_vulnerable_areas']])}

**Climate Risks:**
{chr(10).join(['• ' + risk for risk in risk_info['climate_risks']])}

---

### 💪 **Key Strengths**
{chr(10).join(['• ' + strength for strength in analysis_data['strengths']])}

---

### ⚠️ **Major Vulnerabilities**
{chr(10).join(['• ' + vulnerability for vulnerability in analysis_data['vulnerabilities']])}

---

### 🎯 **Recommendations**

#### Immediate Actions (0-6 months)
{chr(10).join(['• ' + action for action in recommendations['immediate_actions']])}

#### Medium-term Improvements (6 months - 2 years)
{chr(10).join(['• ' + improvement for improvement in recommendations['medium_term_improvements']])}

#### Long-term Strategic Goals (2+ years)
{chr(10).join(['• ' + goal for goal in recommendations['long_term_strategic_goals']])}

---

### 📚 **Analysis Details**
- **Data Sources Considered:** {', '.join(analysis_data['data_sources_considered'])}
- **Confidence Level:** {analysis_data['confidence_level']}
- **Last Updated:** {analysis_data['last_updated']}

---
*Report generated by Gemini AI-powered Disaster Resilience Analyzer*
"""
        
        return report

# Global analyzer instance
analyzer = GeminiDisasterResilienceAnalyzer()

# ==================== AGENTIC AI MODULE ====================
class ClimateAIAgent:
    def __init__(self, region: str, api_key: str):
        self.region = region
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.region_data = self.get_region_data()
        
    def get_region_data(self) -> Dict:
        """Get region-specific climate data and context"""
        region_info = {
            "Delhi": {
                "climate_type": "Semi-arid",
                "major_issues": ["Air pollution", "Heat waves", "Water scarcity", "Urban heat island"],
                "population": "32 million",
                "key_sectors": ["Transportation", "Industry", "Power generation", "Construction"],
                "current_aqi": np.random.randint(150, 400),
                "avg_temp": np.random.randint(35, 45),
                "rainfall_deficit": np.random.randint(20, 60),
                "renewable_percent": 8.5
            },
            "Uttarakhand": {
                "climate_type": "Mountain temperate",
                "major_issues": ["Glacial melting", "Landslides", "Forest fires", "Biodiversity loss"],
                "population": "10.1 million",
                "key_sectors": ["Hydropower", "Tourism", "Agriculture", "Forestry"],
                "current_aqi": np.random.randint(50, 150),
                "avg_temp": np.random.randint(20, 30),
                "rainfall_deficit": np.random.randint(10, 40),
                "renewable_percent": 45.2
            },
            "Assam": {
                "climate_type": "Tropical monsoon",
                "major_issues": ["Flooding", "Riverbank erosion", "Tea plantation impact", "Wetland degradation"],
                "population": "31.2 million",
                "key_sectors": ["Tea industry", "Oil refining", "Agriculture", "Textiles"],
                "current_aqi": np.random.randint(80, 200),
                "avg_temp": np.random.randint(28, 35),
                "rainfall_deficit": np.random.randint(5, 30),
                "renewable_percent": 12.3
            },
            "Andhra Pradesh": {
                "climate_type": "Tropical",
                "major_issues": ["Cyclones", "Drought", "Coastal erosion", "Groundwater depletion"],
                "population": "52.2 million",
                "key_sectors": ["Solar energy", "Agriculture", "Aquaculture", "Pharmaceuticals"],
                "current_aqi": np.random.randint(100, 250),
                "avg_temp": np.random.randint(30, 40),
                "rainfall_deficit": np.random.randint(15, 50),
                "renewable_percent": 22.8
            },
            "Gujarat": {
                "climate_type": "Arid to semi-arid",
                "major_issues": ["Water scarcity", "Coastal salinity", "Industrial pollution", "Heat stress"],
                "population": "60.4 million",
                "key_sectors": ["Chemicals", "Petrochemicals", "Textiles", "Renewable energy"],
                "current_aqi": np.random.randint(120, 300),
                "avg_temp": np.random.randint(32, 42),
                "rainfall_deficit": np.random.randint(25, 65),
                "renewable_percent": 28.5
            }
        }
        return region_info.get(self.region, {})
    
    def generate_response(self, query: str, context_type: str = "general") -> str:
        """Generate AI response based on query and region context"""
        try:
            context = f"""
            You are a specialized Climate AI Agent for {self.region}, India. 
            
            Region Context:
            - Climate Type: {self.region_data.get('climate_type', 'N/A')}
            - Major Climate Issues: {', '.join(self.region_data.get('major_issues', []))}
            - Population: {self.region_data.get('population', 'N/A')}
            - Key Economic Sectors: {', '.join(self.region_data.get('key_sectors', []))}
            - Current AQI: {self.region_data.get('current_aqi', 'N/A')}
            - Average Temperature: {self.region_data.get('avg_temp', 'N/A')}°C
            - Rainfall Deficit: {self.region_data.get('rainfall_deficit', 'N/A')}%
            - Renewable Energy %: {self.region_data.get('renewable_percent', 'N/A')}%
            
            Based on this context and your knowledge of {self.region}, provide specific, actionable, and localized advice.
            Keep responses concise but comprehensive. Include data-driven insights when possible.
            
            User Query: {query}
            """
            
            response = self.model.generate_content(context)
            return response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
    
    def get_mitigation_strategies(self) -> List[str]:
        """Get region-specific mitigation strategies"""
        strategies_map = {
            "Delhi": [
                "Implement odd-even vehicle scheme during high pollution days",
                "Expand metro network and electric bus fleet",
                "Mandate rooftop solar for buildings > 500 sq meters",
                "Create urban forests and vertical gardens",
                "Improve waste-to-energy infrastructure"
            ],
            "Uttarakhand": [
                "Establish glacier monitoring systems",
                "Promote eco-tourism over mass tourism",
                "Implement forest fire early warning systems",
                "Develop micro-hydropower projects",
                "Create wildlife corridors for biodiversity"
            ],
            "Assam": [
                "Build climate-resilient flood management systems",
                "Promote sustainable tea cultivation practices",
                "Restore wetlands and traditional water bodies",
                "Develop early warning systems for floods",
                "Support climate-smart agriculture"
            ],
            "Andhra Pradesh": [
                "Expand solar energy capacity (target 30% by 2030)",
                "Implement coastal zone management plans",
                "Promote drought-resistant crop varieties",
                "Build cyclone shelters in vulnerable areas",
                "Develop water conservation infrastructure"
            ],
            "Gujarat": [
                "Scale up solar and wind energy projects",
                "Implement drip irrigation systems",
                "Develop desalination plants for coastal areas",
                "Promote industrial energy efficiency",
                "Create green industrial corridors"
            ]
        }
        return strategies_map.get(self.region, [])

# Dependency to get API key
def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Google Gemini API key not found in environment variables")
    return api_key

# ==================== Policy FeedBack Loop ====================

# Simulated current policy database (you can replace this with a real DB later)
CURRENT_POLICIES = {
    "Flood": [
        "Evacuation drills every 6 months",
        "Flood forecasting system via SMS alerts"
    ],
    "Earthquake": [
        "Building codes for seismic zones",
        "School-level earthquake drills"
    ],
    "Cyclone": [
        "Cyclone shelters in coastal areas",
        "Disaster communication vans for early warning"
    ]
}

# Simulated external factors (you can make these dynamic later)
REGIONAL_FACTORS = {
    "Kerala": ["High rainfall zone", "Dense population", "Riverine landscape"],
    "Gujarat": ["Seismic zone", "Dry terrain", "Coastal region"],
    "Assam": ["Hilly terrain", "Heavy monsoon", "Poor infrastructure"],
    "Punjab": ["Plains", "Low disaster frequency", "Good road connectivity"]
}


class PolicyRecommendationEngine:
    def generate_evidence_based_policies(self, disaster_patterns: List[Dict]) -> List[Dict]:
        policies = []
        for pattern in disaster_patterns:
            region = pattern.get("region", "Unknown")
            disaster_type = pattern.get("disaster_type", "General")
            frequency = pattern.get("frequency", 1)

            current = CURRENT_POLICIES.get(disaster_type, ["No standard policies found."])
            factors = REGIONAL_FACTORS.get(region, ["General rural area"])

            if frequency >= 5:
                new_policy = (
                    f"Due to frequent {disaster_type} events in {region}, implement AI-based early warning systems, "
                    f"install real-time water level and seismic sensors, and develop mobile-first alert apps for citizens."
                )
            else:
                new_policy = (
                    f"In {region}, occasional {disaster_type} occurrences should be addressed with school safety programs, "
                    f"community awareness sessions, and pre-positioned emergency kits at local panchayats."
                )

            policies.append({
                "region": region,
                "disaster_type": disaster_type,
                "frequency": frequency,
                "current_policies": current,
                "evidence_based_recommendation": new_policy,
                "factors_considered": factors
            })
        return policies


engine = PolicyRecommendationEngine()

# ==================== WEATHER DATA SERVICE ====================
class WeatherService:
    """Weather data service using OpenWeatherMap API"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "https://api.openweathermap.org/data/2.5"  # Use HTTPS
        self.timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
        
        # Validate API key on initialization
        if not self.api_key or self.api_key == "demo_key":
            logger.warning("OpenWeatherMap API key not found. Service will use mock data.")
    
    def _validate_city_name(self, city: str) -> str:
        """Validate and clean city name"""
        if not city or not city.strip():
            raise HTTPException(status_code=400, detail="City name cannot be empty")
        
        # Clean the city name
        cleaned_city = city.strip()
        
        # Basic validation - no special characters except spaces, hyphens, apostrophes
        import re
        if not re.match(r"^[a-zA-Z\s\-']+$", cleaned_city):
            raise HTTPException(status_code=400, detail="Invalid city name format")
        
        if len(cleaned_city) > 100:
            raise HTTPException(status_code=400, detail="City name too long")
            
        return cleaned_city
    
    def _validate_weather_response(self, data: Dict[str, Any]) -> None:
        """Validate the API response structure"""
        required_fields = ["main", "wind", "name"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate main weather data
        main_fields = ["temp", "humidity", "pressure"]
        for field in main_fields:
            if field not in data["main"]:
                raise ValueError(f"Missing weather field: main.{field}")
    
    def _extract_weather_data(self, data: Dict[str, Any], city: str) -> WeatherData:
        """Extract and validate weather data from API response"""
        try:
            # Extract main weather data
            main = data["main"]
            wind = data.get("wind", {})
            weather = data.get("weather", [{}])[0]
            
            # Extract precipitation data (rain or snow)
            precipitation = 0.0
            if "rain" in data:
                precipitation = data["rain"].get("1h", 0.0)
            elif "snow" in data:
                precipitation = data["snow"].get("1h", 0.0)
            
            return WeatherData(
                temperature=float(main["temp"]),
                humidity=float(main["humidity"]),
                pressure=float(main["pressure"]),
                wind_speed=float(wind.get("speed", 0.0)),
                precipitation=float(precipitation),
                location=data.get("name", city),
                timestamp=datetime.now(),
                weather_description=weather.get("description"),
                feels_like=float(main.get("feels_like", main["temp"]))
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error extracting weather data: {e}")
            raise ValueError(f"Invalid weather data format: {e}")
    
    async def _fetch_from_api(self, city: str) -> Dict[str, Any]:
        """Fetch weather data from OpenWeatherMap API"""
        url = f"{self.base_url}/weather"
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    # Handle different HTTP status codes
                    if response.status == 401:
                        logger.error("Invalid API key")
                        raise HTTPException(
                            status_code=500, 
                            detail="Weather service authentication failed"
                        )
                    elif response.status == 404:
                        logger.warning(f"City not found: {city}")
                        raise HTTPException(
                            status_code=404, 
                            detail=f"City '{city}' not found"
                        )
                    elif response.status == 429:
                        logger.warning("API rate limit exceeded")
                        raise HTTPException(
                            status_code=429, 
                            detail="Weather service rate limit exceeded. Please try again later."
                        )
                    elif response.status != 200:
                        logger.error(f"API request failed with status {response.status}")
                        raise HTTPException(
                            status_code=500, 
                            detail="Weather service temporarily unavailable"
                        )
                    
                    return await response.json()
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error while fetching weather data: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Unable to connect to weather service"
            )
        except asyncio.TimeoutError:
            logger.error("Request timeout while fetching weather data")
            raise HTTPException(
                status_code=504, 
                detail="Weather service request timeout"
            )
    
    def _get_mock_data(self, city: str) -> WeatherData:
        """Generate mock weather data for testing"""
        logger.info(f"Returning mock weather data for {city}")
        return WeatherData(
            temperature=np.random.uniform(20, 35),
            humidity=np.random.uniform(40, 80),
            pressure=np.random.uniform(1000, 1020),
            wind_speed=np.random.uniform(5, 15),
            precipitation=np.random.uniform(0, 10),
            location=city,
            timestamp=datetime.now(),
            weather_description="Mock weather data",
            feels_like=np.random.uniform(22, 37)
        )
    
    async def get_weather_data(self, city: str) -> WeatherData:
        """
        Fetch weather data for a given city
        
        Args:
            city: Name of the city
            
        Returns:
            WeatherData object containing weather information
            
        Raises:
            HTTPException: For various error conditions
        """
        # Validate city name
        cleaned_city = self._validate_city_name(city)
        
        # If no API key, return mock data
        if not self.api_key or self.api_key == "demo_key":
            return self._get_mock_data(cleaned_city)
        
        try:
            # Fetch data from API
            api_data = await self._fetch_from_api(cleaned_city)
            
            # Validate response structure
            self._validate_weather_response(api_data)
            
            # Extract and return weather data
            weather_data = self._extract_weather_data(api_data, cleaned_city)
            
            logger.info(f"Successfully fetched weather data for {weather_data.location}")
            return weather_data
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching weather data for {city}: {e}")
            raise HTTPException(
                status_code=500, 
                detail="An unexpected error occurred while fetching weather data"
            )

# ==================== GLOBAL INSTANCES ====================

weather_service = WeatherService()
blockchain = GovernanceBlockchain()
quantum_optimizer = QuantumResourceOptimizer()



# In-memory storage (use database in production)
citizen_reports = []
disaster_alerts = []
resource_optimizations = []

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ClimaX Backend - Climate Resilience OS",
        "version": "1.0.0",
        "modules": ["AI Agents", "Quantum Optimizer", "Blockchain", "RAG", "Weather Service"]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    api_key_status = "configured" if weather_service.api_key and weather_service.api_key != "demo_key" else "missing"
    
    return {
        "status": "healthy",
        "service": "weather-api",
        "api_key_status": api_key_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/weather/{city}")
async def get_weather(city: str):
    """
    Get weather data for a city
    
    Args:
        city: Name of the city (path parameter)
        
    Returns:
        JSON response with weather data
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        weather_data = await weather_service.get_weather_data(city)
        
        return {
            "success": True,
            "data": {
                "temperature": round(weather_data.temperature, 1),
                "humidity": weather_data.humidity,
                "pressure": weather_data.pressure,
                "wind_speed": round(weather_data.wind_speed, 1),
                "precipitation": round(weather_data.precipitation, 2),
                "location": weather_data.location,
                "timestamp": weather_data.timestamp.isoformat(),
                "weather_description": weather_data.weather_description,
                "feels_like": round(weather_data.feels_like, 1) if weather_data.feels_like else None
            }
        }
    except HTTPException:
        # Re-raise FastAPI HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in weather endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom exception handler for better error responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.status_code,
                "message": exc.detail
            }
        }
    )

@app.post("/citizen-report")
async def submit_citizen_report(report: CitizenReport):
    """Submit citizen disaster report"""
    report.id = str(uuid.uuid4())
    report.timestamp = datetime.now()

    # Simple verification simulation (in real app, use AI for verification)
    if len(report.description) > 10:
        report.verified = True

    citizen_reports.append(report)
    logger.info(f"Citizen report submitted: {report.id}")

    return {"status": "success", "report_id": report.id, "verified": report.verified}

@app.get("/citizen-reports")
async def get_citizen_reports(limit: int = 10):
    """Get recent citizen reports"""
    # Convert to dict format for JSON serialization
    reports_data = []
    for report in citizen_reports[-limit:]:
        report_dict = {
            "id": report.id,
            "location": report.location,
            "disaster_type": report.disaster_type,
            "severity": report.severity,
            "description": report.description,
            "image_url": report.image_url,
            "timestamp": report.timestamp.isoformat() if isinstance(report.timestamp, datetime) else report.timestamp,
            "verified": report.verified
        }
        reports_data.append(report_dict)
    return reports_data

@app.post("/optimize-resources")
async def optimize_resources(request: Dict[str, Any]):
    """Optimize resource allocation using quantum computing"""
    regions = request.get("regions", [])
    resources = request.get("resources", {})
    demands = request.get("demands", {})

    optimization_result = await quantum_optimizer.optimize_resources(regions, resources, demands)

    # Store optimization result
    resource_opt = ResourceOptimization(
        region=",".join(regions),
        resources=resources,
        demand=demands,
        optimization_result=optimization_result,
        quantum_runtime=optimization_result["quantum_runtime"]
    )

    resource_optimizations.append(resource_opt)

    return optimization_result

@app.get("/alerts")
async def get_alerts(limit: int = 10):
    """Get recent disaster alerts"""
    # Convert to dict format for JSON serialization
    alerts_data = []
    for alert in disaster_alerts[-limit:]:
        alert_dict = {
            "id": alert.id,
            "region": alert.region,
            "disaster_type": alert.disaster_type,
            "alert_level": alert.alert_level,
            "description": alert.description,
            "affected_area": alert.affected_area,
            "evacuation_routes": alert.evacuation_routes,
            "resources_needed": alert.resources_needed,
            "timestamp": alert.timestamp.isoformat() if isinstance(alert.timestamp, datetime) else alert.timestamp,
            "blockchain_hash": alert.blockchain_hash
        }
        alerts_data.append(alert_dict)
    return alerts_data

@app.get("/alerts/{region}")
async def get_regional_alerts(region: str, limit: int = 10):
    """Get alerts for specific region"""
    regional_alerts = [alert for alert in disaster_alerts if alert.region.lower() == region.lower()]
    
    # Convert to dict format for JSON serialization
    alerts_data = []
    for alert in regional_alerts[-limit:]:
        alert_dict = {
            "id": alert.id,
            "region": alert.region,
            "disaster_type": alert.disaster_type,
            "alert_level": alert.alert_level,
            "description": alert.description,
            "affected_area": alert.affected_area,
            "evacuation_routes": alert.evacuation_routes,
            "resources_needed": alert.resources_needed,
            "timestamp": alert.timestamp.isoformat() if isinstance(alert.timestamp, datetime) else alert.timestamp,
            "blockchain_hash": alert.blockchain_hash
        }
        alerts_data.append(alert_dict)
    return alerts_data

@app.post("/test/send-alert-sms")
async def send_latest_alert_sms():
    """Send the most recent generated alert as SMS via Twilio"""
    try:
        if not disaster_alerts:
            raise HTTPException(status_code=404, detail="No alerts available to send.")

        # Get the latest alert
        latest_alert = disaster_alerts[-1]

        # Prepare the alert message
        sms_message = (
            f"🚨 Disaster Alert 🚨\n"
            f"Region: {latest_alert.region}\n"
            f"Type: {latest_alert.disaster_type}\n"
            
            f"Affected Area: {latest_alert.affected_area}\n"
                    )

        # Twilio client setup
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_PHONE_NUMBER")
        to_number = os.getenv("TARGET_PHONE_NUMBER")

        if not all([account_sid, auth_token, from_number, to_number]):
            raise HTTPException(status_code=500, detail="Twilio credentials not configured properly.")

        client = Client(account_sid, auth_token)

        # Send SMS
        message = client.messages.create(
            body=sms_message,
            from_=from_number,
            to=to_number
        )

        logger.info(f"SMS sent successfully. SID: {message.sid}")

        return {"success": True, "message_sid": message.sid, "to": to_number}

    except Exception as e:
        logger.error(f"Failed to send SMS alert: {e}")
        raise HTTPException(status_code=500, detail=f"SMS sending failed: {str(e)}")

@app.get("/knowledge/query")
async def query_knowledge(query: str, disaster_type: Optional[str] = None):
    """Query climate knowledge base using RAG"""
    kb = ClimateKnowledgeBase()
    results = kb.query_knowledge(query, disaster_type)
    return {"query": query, "results": results}

@app.post("/test/generate-alerts")
async def generate_test_alerts():
    """Generate test alerts for demonstration"""
    try:
        logger.info("Starting test alert generation...")
        test_regions = ["delhi", "mumbai", "bangalore"]
        generated_alerts = []

        for region in test_regions:
            try:
                logger.info(f"Generating alert for {region}...")
                weather_data = await weather_service.get_weather_data(region)
                agent = ClimateAIAgent[region]

                # Force high threat for demo
                weather_data.temperature = 42
                weather_data.precipitation = 60

                threat_analysis = agent.analyze_threat(weather_data)
                alert = agent.generate_alert(threat_analysis)

                if alert:
                    blockchain_hash = blockchain.add_alert_to_chain(alert)
                    alert.blockchain_hash = blockchain_hash
                    disaster_alerts.append(alert)
                    generated_alerts.append(alert)
                    logger.info(f"Generated alert for {region}: {alert.id}")

            except Exception as e:
                logger.error(f"Error generating alert for {region}: {e}")
                continue

        logger.info(f"Successfully generated {len(generated_alerts)} alerts")
        
        # Convert alerts to dict format for JSON response
        alerts_data = []
        for alert in generated_alerts:
            alert_dict = {
                "id": alert.id,
                "region": alert.region,
                "disaster_type": alert.disaster_type,
                "alert_level": alert.alert_level,
                "description": alert.description,
                "affected_area": alert.affected_area,
                "evacuation_routes": alert.evacuation_routes,
                "resources_needed": alert.resources_needed,
                "timestamp": alert.timestamp.isoformat() if isinstance(alert.timestamp, datetime) else alert.timestamp,
                "blockchain_hash": alert.blockchain_hash
            }
            alerts_data.append(alert_dict)

        return {"generated_alerts": len(generated_alerts), "alerts": alerts_data}

    except Exception as e:
        logger.error(f"Error in generate_test_alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate test alerts: {str(e)}")
    
##chat bot

async def root():
    """Health check endpoint"""
    return {
        "message": "Disaster Response Bot API is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }
@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(chat_input: ChatRequest):
    """Chat with the AI assistant"""
    try:
        bot = get_bot()
        response = bot.get_ai_response(chat_input.question, chat_input.language)
        
        return ChatResponse(
            response=response,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/weather", response_model=WeatherResponse)
async def get_weather(request: WeatherRequest):
    """Get weather information for a location"""
    try:
        bot = get_bot()
        weather_data = bot.get_weather(request.location)
        
        return WeatherResponse(
            weather_info=weather_data["weather_info"],
            success=weather_data["success"]
        )
    except Exception as e:
        logger.error(f"Weather error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching weather: {str(e)}")

@app.get("/emergency-contacts/{language}")
async def get_emergency_contacts(language: str = "en"):
    """Get emergency contacts in specified language"""
    try:
        bot = get_bot()
        contacts = bot.get_emergency_contacts(language)
        
        return {
            "contacts": contacts,
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Emergency contacts error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching contacts: {str(e)}")

@app.get("/disaster-guide/{disaster_type}")
async def get_disaster_guide(disaster_type: str, language: str = "en"):
    """Get disaster safety guide for specific disaster type"""
    try:
        bot = get_bot()
        guide = bot.get_disaster_guide(disaster_type.lower(), language)
        
        if guide is None:
            raise HTTPException(status_code=404, detail=f"No guide found for disaster type: {disaster_type}")
        
        return {
            "disaster_type": disaster_type,
            "language": language,
            "guide": guide,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Disaster guide error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching guide: {str(e)}")

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback"""
    try:
        bot = get_bot()
        result = bot.save_feedback(
            request.safety_status,
            request.govt_rating,
            request.feedback,
            request.language,
            request.location or "Unknown"
        )
        
        return FeedbackResponse(
            success=result["success"],
            message=result["message"]
        )
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")

@app.get("/feedback/summary")
async def get_feedback_summary():
    """Get feedback analytics summary (admin endpoint)"""
    try:
        bot = get_bot()
        summary = bot.get_feedback_summary()
        
        if not summary["success"]:
            raise HTTPException(status_code=404, detail=summary["message"])
        
        return {
            "summary": summary["data"],
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching summary: {str(e)}")

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    try:
        bot = get_bot()
        return {
            "languages": bot.languages,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Languages error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching languages: {str(e)}")

# Health check endpoints
@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        bot = get_bot()
        return {
            "status": "healthy",
            "services": {
                "gemini": "connected" if bot.gemini_model else "disconnected",
                "granite": "connected" if bot.granite_llm else "disconnected",
                "weather_api": "configured" if bot.weather_api_key else "not configured"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Update your TTS endpoint
@app.post("/api/text-to-speech")
async def text_to_speech(request: TTSRequest):
    try:
        # Generate unique filename
        audio_filename = f"tts_{uuid.uuid4().hex}.wav"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        
        # Your existing TTS generation code, but save to audio_path instead of temp file
        voice_processor = VoiceProcessor()
        
        # Generate audio and save to the permanent location
        success = voice_processor.text_to_speech(
            text=request.text,
            language=request.language,
            output_file=audio_path,  # Save directly to permanent location
            speed=request.speed
        )
        
        if success and os.path.exists(audio_path):
            # Return the URL path that can be accessed by the frontend
            audio_url = f"/audio/{audio_filename}"
            
            return {
                "success": True,
                "message": "TTS generated successfully",
                "audio_url": audio_url,
                "audio_filename": audio_filename
            }
        else:
            raise HTTPException(status_code=500, detail="TTS generation failed")
            
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve audio files"""
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Optional: Cleanup endpoint to remove old audio files
@app.delete("/api/cleanup-audio")
async def cleanup_audio():
    """Clean up old audio files"""
    try:
        import time
        current_time = time.time()
        deleted_count = 0
        
        for filename in os.listdir(AUDIO_DIR):
            file_path = os.path.join(AUDIO_DIR, filename)
            # Delete files older than 1 hour
            if os.path.isfile(file_path) and (current_time - os.path.getctime(file_path)) > 3600:
                os.remove(file_path)
                deleted_count += 1
        
        return {"success": True, "deleted_files": deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
    
@app.post("/api/speech-to-text", response_model=STTResponse)
async def speech_to_text(
    audio: UploadFile = File(...),
    language: str = Form("en")
):
    """
    Convert speech to text
    
    - **audio**: Audio file (WAV, MP3, etc.)
    - **language**: Language code for recognition
    """
    try:
        # Validate file type
        allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/webm', 'audio/ogg']
        if audio.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format. Allowed: {', '.join(allowed_types)}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio.read()
            await aiofiles.open(temp_file.name, 'wb').write(content)
            temp_path = temp_file.name
        
        # Run STT in thread pool
        loop = asyncio.get_event_loop()
        recognized_text = await loop.run_in_executor(
            executor,
            perform_speech_recognition,
            temp_path,
            language
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        if recognized_text:
            return STTResponse(success=True, text=recognized_text)
        else:
            return STTResponse(success=False, error="Could not recognize speech")
            
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"STT Error: {str(e)}")

def perform_speech_recognition(audio_path: str, language: str) -> Optional[str]:
    """Perform speech recognition in a separate thread"""
    try:
        recognizer = sr.Recognizer()
        
        # Language mapping for Google Speech Recognition
        language_map = {
            'en': 'en-US',
            'hi': 'hi-IN',
            'kn': 'kn-IN',
            'te': 'te-IN',
            'ta': 'ta-IN',
            'bn': 'bn-IN'
        }
        
        with sr.AudioFile(audio_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
        
        # Recognize speech
        text = recognizer.recognize_google(
            audio_data,
            language=language_map.get(language, 'en-US')
        )
        
        return text
        
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        raise Exception(f"Speech recognition service error: {e}")
    except Exception as e:
        raise Exception(f"Recognition error: {e}")

@app.post("/api/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    language: str = Form("en"),
    location: str = Form("")
):
    """
    Complete voice chat pipeline: STT -> Process -> TTS
    
    - **audio**: User's voice input
    - **language**: Language for processing
    - **location**: User location for weather/local info
    """
    try:
        # Step 1: Convert speech to text
        stt_response = await speech_to_text(audio, language)
        
        if not stt_response.success:
            raise HTTPException(status_code=400, detail="Could not understand speech")
        
        user_text = stt_response.text
        
        # Step 2: Get bot response
        if location and any(word in user_text.lower() for word in ['weather', 'temperature', 'climate']):
            weather_info = disaster_bot.get_weather(location)
            bot_response = weather_info['weather_info']
        else:
            bot_response = disaster_bot.get_ai_response(user_text, language)
        
        # Step 3: Convert response to speech
        tts_request = TTSRequest(text=bot_response, language=language)
        tts_response = await text_to_speech(tts_request)
        
        return {
            "success": True,
            "user_text": user_text,
            "bot_response": bot_response,
            "audio_url": tts_response.audio_url if tts_response.success else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice chat error: {str(e)}")

@app.get("/api/voice/languages")
async def get_supported_languages():
    """Get list of supported languages for voice features"""
    return {
        "languages": [
            {"code": "en", "name": "English", "flag": "🇺🇸"},
            {"code": "hi", "name": "हिंदी", "flag": "🇮🇳"},
            {"code": "kn", "name": "ಕನ್ನಡ", "flag": "🇮🇳"},
            {"code": "te", "name": "తెలుగు", "flag": "🇮🇳"},
            {"code": "ta", "name": "தமிழ்", "flag": "🇮🇳"},
            {"code": "bn", "name": "বাংলা", "flag": "🇧🇩"}
        ],
        "tts_supported": True,
        "stt_supported": True
    }


#Agentic Ai

@app.get("/regions")
async def get_regions():
    """Get list of available regions"""
    return {"regions": ["Delhi", "Uttarakhand", "Assam", "Andhra Pradesh", "Gujarat"]}

@app.get("/region/{region}/data", response_model=RegionData)
async def get_region_data(region: str, api_key: str = Depends(get_api_key)):
    """Get region-specific data"""
    try:
        agent = ClimateAIAgent(region, api_key)
        return RegionData(**agent.region_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/region/{region}/metrics", response_model=ClimateMetrics)
async def get_climate_metrics(region: str, api_key: str = Depends(get_api_key)):
    """Get current climate metrics for a region"""
    try:
        agent = ClimateAIAgent(region, api_key)
        return ClimateMetrics(
            temperature=agent.region_data.get('avg_temp', 0),
            aqi=agent.region_data.get('current_aqi', 0),
            rainfall_deficit=agent.region_data.get('rainfall_deficit', 0),
            renewable_percent=agent.region_data.get('renewable_percent', 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/region/{region}/trends", response_model=TrendData)
async def get_climate_trends(region: str):
    """Get climate trend data for charts"""
    try:
        # Generate sample data for demonstration
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
        
        trend_data = TrendData(
            dates=[date.strftime('%Y-%m-%d') for date in dates],
            temperatures=(np.random.normal(30, 5, len(dates)) + np.linspace(0, 3, len(dates))).tolist(),
            aqi_values=(np.random.normal(150, 50, len(dates)) + np.sin(np.arange(len(dates)) * 0.5) * 30).tolist()
        )
        
        return trend_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_agent(request: ChatRequest, api_key: str = Depends(get_api_key)):
    """Chat with the climate AI agent"""
    try:
        agent = ClimateAIAgent(request.region, api_key)
        response = agent.generate_response(request.query, request.context_type)
        return {
            "query": request.query,
            "response": response,
            "region": request.region,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis")
async def generate_analysis(request: AnalysisRequest, api_key: str = Depends(get_api_key)):
    """Generate climate analysis"""
    try:
        agent = ClimateAIAgent(request.region, api_key)
        query = f"Provide detailed {request.analysis_type.lower()} for {request.region} considering current climate trends and future projections"
        analysis = agent.generate_response(query, "analysis")
        
        return {
            "analysis_type": request.analysis_type,
            "region": request.region,
            "analysis": analysis,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/region/{region}/strategies")
async def get_mitigation_strategies(region: str, api_key: str = Depends(get_api_key)):
    """Get mitigation strategies for a region"""
    try:
        agent = ClimateAIAgent(region, api_key)
        strategies = agent.get_mitigation_strategies()
        return {
            "region": region,
            "strategies": strategies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/strategies/custom")
async def generate_custom_strategy(request: StrategyRequest, api_key: str = Depends(get_api_key)):
    """Generate custom strategy for specific sector"""
    try:
        agent = ClimateAIAgent(request.region, api_key)
        query = f"Suggest specific climate mitigation strategies for the {request.sector.lower()} sector in {request.region}"
        strategy = agent.generate_response(query, "strategy")
        
        return {
            "region": request.region,
            "sector": request.sector,
            "strategy": strategy,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports")
async def generate_report(request: ReportRequest, api_key: str = Depends(get_api_key)):
    """Generate comprehensive climate report"""
    try:
        agent = ClimateAIAgent(request.region, api_key)
        query = f"Generate a comprehensive {request.report_type.lower()} for climate change mitigation in {request.region} focusing on {request.time_period.lower()}. Include specific data, recommendations, and implementation timeline."
        report = agent.generate_response(query, "report")
        
        return {
            "report_type": request.report_type,
            "region": request.region,
            "time_period": request.time_period,
            "report": report,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/comparison")
async def get_region_comparison():
    """Get comparative analysis data for all regions"""
    try:
        regions = ["Delhi", "Uttarakhand", "Assam", "Andhra Pradesh", "Gujarat"]
        comparison_data = []
        
        for region in regions:
            comparison_data.append({
                "region": region,
                "temperature_risk": np.random.randint(1, 10),
                "water_stress": np.random.randint(1, 10),
                "air_quality": np.random.randint(1, 10),
                "renewable_potential": np.random.randint(1, 10)
            })
        
        return {"comparison_data": comparison_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#blockchain

@app.post("/citizen/register")
def register_citizen(data: CitizenRegistration):
    if data.citizen_id in citizens_db:
        raise HTTPException(status_code=400, detail="Citizen already registered")

    # Generate RSA keypair
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    # Serialize public key to PEM format
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")

    citizens_db[data.citizen_id] = {
        "name": data.name,
        "public_key": public_pem
        # You could also store private key securely (not shown)
    }

    return {
        "message": "Citizen registered successfully",
        "citizen_id": data.citizen_id,
        "public_key": public_pem
    }

@app.get("/citizen/public-key/{citizen_id}")
def get_public_key(citizen_id: str):
    citizen = citizens_db.get(citizen_id)
    if not citizen:
        raise HTTPException(status_code=404, detail="Citizen not found")

    return {
        "citizen_id": citizen_id,
        "public_key": citizen["public_key"]
    }

@app.post("/organization/register")
def register_organization(data: OrganizationRegistration):
    org_name = data.organization_name.lower()

    if org_name in organizations_db:
        raise HTTPException(status_code=400, detail="Organization already registered")

    # Generate RSA keypair
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    # Serialize public key
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")

    organizations_db[org_name] = {
        "public_key": public_pem
        # Store private key securely if needed
    }

    return {
        "message": "Organization registered successfully",
        "organization_name": org_name,
        "public_key": public_pem  
    }

@app.get("/organization/public-key/{organization_name}")
def get_organization_public_key(organization_name: str):
    org_name = organization_name.lower()

    organization = organizations_db.get(org_name)
    if not organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    return {
        "organization_name": org_name,
        "public_key": organization["public_key"]
    }

# 1️⃣ Submit a new alert (signed)
@app.post("/alerts/submit")
def submit_alert(alert: AlertSubmission):
    org_name = alert.organization_name.lower()
    org = organizations_db.get(org_name)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not registered")

    # Get public key object
    public_key_pem = org["public_key"]
    public_key = serialization.load_pem_public_key(public_key_pem.encode())

    # Create the message to verify
    message = f"{alert.organization_name}|{alert.alert_type}|{alert.affected_area}|{alert.alert_message}|{alert.timestamp}"
    message_bytes = message.encode()
    signature_bytes = base64.b64decode(alert.signature)

    try:
        public_key.verify(
            signature_bytes,
            message_bytes,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
    except InvalidSignature:
        raise HTTPException(status_code=400, detail="Invalid digital signature")

    alert_data = {
        "organization_name": org_name,
        "alert_type": alert.alert_type,
        "alert_message": alert.alert_message,
        "affected_area": alert.affected_area,
        "timestamp": alert.timestamp,
        "signature": alert.signature,
        "verified": True
    }
    alerts_db.append(alert_data)

    return {
        "message": "Alert submitted and verified successfully",
        "alert": alert_data
    }

# 2️⃣ Get all active alerts
@app.get("/alerts/active")
def get_active_alerts():
    return [alert for alert in alerts_db if alert["verified"]]

# 3️⃣ Verify an alert (explicit check)
@app.post("/alerts/verify")
def verify_alert(alert: AlertVerificationRequest):
    org_name = alert.organization_name.lower()
    org = organizations_db.get(org_name)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    public_key_pem = org["public_key"]
    public_key = serialization.load_pem_public_key(public_key_pem.encode())

    message = f"{alert.organization_name}|{alert.alert_type}|{alert.affected_area}|{alert.alert_message}|{alert.timestamp}"
    message_bytes = message.encode()
    signature_bytes = base64.b64decode(alert.signature)

    try:
        public_key.verify(
            signature_bytes,
            message_bytes,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return {"authentic": True, "message": "Alert signature is valid"}
    except InvalidSignature:
        return {"authentic": False, "message": "Invalid signature. Alert may be forged"}

# Submit feedback (signed or anonymous)
@app.post("/feedback/submit")
def submit_feedback(feedback: Feedback):
    if feedback.citizen_id and feedback.signature:
        # Signed feedback – verify it
        public_key = citizen_public_keys.get(feedback.citizen_id)
        if not public_key:
            raise HTTPException(status_code=404, detail="Citizen not found")

        try:
            public_key.verify(
                base64.b64decode(feedback.signature),
                feedback.feedback.encode(),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
        except InvalidSignature:
            raise HTTPException(status_code=400, detail="Invalid signature")

    feedback_db.append(feedback)
    return {"message": "Feedback submitted successfully"}


# Get all feedbacks (admin only)
@app.get("/feedback/all")
def get_all_feedbacks():
    return feedback_db


# Get feedback stats (simple analytics)
@app.get("/feedback/analytics")
def feedback_analytics():
    total_feedbacks = len(feedback_db)
    signed = sum(1 for f in feedback_db if f.citizen_id)
    anonymous = total_feedbacks - signed

    # Example sentiment-based keyword chart (mocked)
    keywords = ["good", "bad", "urgent", "safe"]
    keyword_counts = Counter()
    for entry in feedback_db:
        for word in keywords:
            if word in entry.feedback.lower():
                keyword_counts[word] += 1

    return {
        "total_feedbacks": total_feedbacks,
        "signed": signed,
        "anonymous": anonymous,
        "keyword_mentions": keyword_counts
    }

# Log a new government action
@app.post("/gov/action/log")
def log_gov_action(action: GovAction):
    gov_actions_db.append(action)
    return {"message": "Action logged successfully", "action": action}

# View all government actions
@app.get("/gov/action/history")
def get_action_history():
    return {"total_actions": len(gov_actions_db), "actions": gov_actions_db}

@app.get("/blockchain/view")
def view_blockchain():
    chain = [
        {
            "index": b.index,
            "timestamp": b.timestamp,
            "data": b.transactions,
            "previous_hash": b.previous_hash,
            "hash": b.hash
        }
        for b in blockchain.chain
    ]
    return JSONResponse(content={"length": len(chain), "chain": chain})

@app.post("/blockchain/mine")
def mine_block(new_data: NewBlockData):
    previous_block = blockchain.chain[-1]
    new_block = Block(
        index=previous_block.index + 1,
        timestamp=time.time(),
        transactions=new_data.data,
        previous_hash=previous_block.hash
    )
    blockchain.chain.append(new_block)
    return {"message": "Block mined successfully", "block": new_block.__dict__}

@app.get("/blockchain/verify")
def verify_blockchain():
    chain = blockchain.chain  # ✅ Access the internal list of blocks

    
    for i in range(1, len(blockchain.chain)):
        prev = blockchain[i - 1].chain
        curr = blockchain[i].chain

        if curr.previous_hash != prev.hash:
            return {"valid": False, "error": f"Invalid link between block {i-1} and {i}"}
        
        if curr.hash != curr.calculate_hash():
            return {"valid": False, "error": f"Hash mismatch in block {i}"}

    return {"valid": True, "message": "Blockchain is valid"}

@app.post("/blockchain/tamper")
def tamper_blockchain(tamper: TamperRequest):
    if 0 < tamper.block_index < len(blockchain):
        block = blockchain[tamper.block_index]
        block.data = tamper.new_data
        # Do NOT recalculate hash (to simulate tampering)
        return {"message": f"Block {tamper.block_index} tampered successfully"}
    else:
        return {"error": "Invalid block index"}

@app.get("/blockchain/status")
def blockchain_status():
    for i in range(1, len(blockchain.chain)):
        prev = blockchain[i - 1].chain
        curr = blockchain[i].chain

        if curr.previous_hash != prev.hash or curr.hash != curr.calculate_hash():
            return {"status": "invalid", "problem_block": i}
    
    return {"status": "valid", "message": "Blockchain integrity intact"}

#Resilience Score

@app.get("/", response_model=StatusResponse)
async def root():
    """Health check endpoint"""
    return StatusResponse(
        status="healthy",
        message="Disaster Resilience Analyzer API is running",
        timestamp=datetime.now().isoformat()
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_location(request: LocationAnalysisRequest):
    """Analyze disaster resilience for a given location"""
    try:
        analysis_data = await analyzer.get_comprehensive_resilience_analysis(
            request.location, 
            request.gemini_api_key
        )
        return AnalysisResponse(**analysis_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_location: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/dashboard")
async def get_dashboard(request: LocationAnalysisRequest):
    """Get dashboard visualizations for a location analysis"""
    try:
        # First perform the analysis
        analysis_data = await analyzer.get_comprehensive_resilience_analysis(
            request.location, 
            request.gemini_api_key
        )
        
        # Then create the dashboard
        dashboard_data = analyzer.create_resilience_dashboard(analysis_data)
        
        return {
            "analysis": AnalysisResponse(**analysis_data),
            "dashboard": dashboard_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard creation failed: {str(e)}")

@app.post("/report")
async def get_detailed_report(request: LocationAnalysisRequest):
    """Get detailed markdown report for a location analysis"""
    try:
        # First perform the analysis
        analysis_data = await analyzer.get_comprehensive_resilience_analysis(
            request.location, 
            request.gemini_api_key
        )
        
        # Then create the report
        detailed_report = analyzer.create_detailed_report(analysis_data)
        
        return {
            "analysis": AnalysisResponse(**analysis_data),
            "report": detailed_report
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_detailed_report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/history", response_model=HistoryResponse)
async def get_analysis_history():
    """Get analysis history"""
    try:
        analyses = []
        for analysis in analyzer.analysis_history[-10:]:  # Last 10 analyses
            analyses.append(HistoryItem(
                location_name=analysis['location_info']['full_name'],
                overall_score=analysis['overall_resilience_score'],
                risk_level=analysis['risk_assessment']['risk_level'],
                timestamp=analysis['timestamp']
            ))
        
        return HistoryResponse(
            analyses=analyses,
            total_count=len(analyzer.analysis_history)
        )
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis history")

@app.delete("/history")
async def clear_analysis_history():
    """Clear analysis history"""
    try:
        analyzer.analysis_history.clear()
        return StatusResponse(
            status="success",
            message="Analysis history cleared",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear analysis history")

@app.get("/categories")
async def get_categories():
    """Get available resilience categories"""
    return {
        "categories": analyzer.categories,
        "description": "Six key categories used for disaster resilience assessment"
    }

#policy feedback loop

@app.post("/generate-evidence-based-policies")
def generate_policies(disaster_patterns: List[DisasterPattern]):
    input_data = [pattern.dict() for pattern in disaster_patterns]
    return {"detailed_policy_report": engine.generate_evidence_based_policies(input_data)}

# ==================== STARTUP EVENT ==================== 

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("🚀 ClimaX Backend Starting Up...")
    logger.info("🤖 AI Agents: Initialized")
    logger.info("⚛️ Quantum Optimizer: Ready")
    logger.info("🔗 Blockchain: Genesis block created")
    logger.info("🧠 RAG System: Knowledge base loaded")
    logger.info("🌤️ Weather Service: Connected")
    logger.info("✅ ClimaX Backend Ready!")
    try:
        get_bot()
        logger.info("Disaster Response Bot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)