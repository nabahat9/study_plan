from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid
import traceback

from models import StudyPlanRequest, StudyPlanResponse
from study_planner import StudyPlanner

# Initialize FastAPI
app = FastAPI(
    title="Study Plan Suggester API",
    description="Generate personalized study plans using ECLAT algorithm",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global planner instance
planner = None

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global planner
    try:
        print("🚀 Starting Study Planner initialization...")
        
        # Initialize planner
        planner = StudyPlanner("data/study_plans.csv")
        
        # Load and preprocess data
        planner.load_and_preprocess()
        
        # Run ECLAT algorithm
        planner.run_eclat(min_support=0.15, min_confidence=0.6)
        
        print("✅ Study Planner initialized successfully!")
        print(f"   - Loaded {len(planner.df) if planner.df is not None else 0} data points")
        print(f"   - Generated {len(planner.rules) if planner.rules is not None else 0} association rules")
        
    except Exception as e:
        print(f"❌ Failed to initialize Study Planner: {e}")
        traceback.print_exc()
        planner = None

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Study Plan Suggester API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "POST /generate-plan": "Generate study plan",
            "GET /rules": "Get ECLAT association rules",
            "GET /status": "Get planner status"
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if planner is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "planner_initialized": planner is not None
    }

@app.get("/status")
async def get_status():
    """Get detailed planner status"""
    if planner is None:
        raise HTTPException(status_code=503, detail="Planner not initialized")
    
    status = {
        "planner_initialized": True,
        "data_loaded": planner.df is not None,
        "rules_generated": planner.rules is not None,
        "data_shape": planner.df.shape if planner.df is not None else None,
        "rules_count": len(planner.rules) if planner.rules is not None else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    return status

@app.get("/rules")
async def get_rules(limit: int = 20):
    """Get ECLAT association rules"""
    if planner is None:
        raise HTTPException(status_code=503, detail="Planner not initialized")
    
    try:
        rules = planner.get_rules(limit)
        return {
            "success": True,
            "count": len(rules),
            "rules": rules
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-plan", response_model=StudyPlanResponse)
async def generate_study_plan(request: StudyPlanRequest):
    """Generate a personalized study plan"""
    if planner is None:
        raise HTTPException(status_code=503, detail="Planner not initialized")
    
    try:
        print(f"📥 Received request for: {request.goal}")
        
        # Convert request to dict
        user_input = request.dict()
        user_input['deadline'] = str(user_input['deadline'])
        
        print(f"🔧 User input: {user_input}")
        
        # Generate study plan
        daily_plans = planner.generate_study_plan(user_input)
        
        if not daily_plans:
            raise HTTPException(status_code=400, detail="Could not generate any study plans")
        
        # Calculate totals
        total_study_minutes = sum(p['total_study_minutes'] for p in daily_plans)
        total_hours = total_study_minutes / 60
        
        # Create response
        response = StudyPlanResponse(
            success=True,
            message="Study plan generated successfully",
            plan_id=str(uuid.uuid4())[:8],
            total_days=len(daily_plans),
            total_hours=round(total_hours, 2),
            daily_plans=daily_plans,
            generated_at=datetime.now().isoformat()
        )
        
        print(f"✅ Generated plan with {len(daily_plans)} days, {total_hours:.1f} hours")
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"🔥 Error generating plan: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/test-plan")
async def test_plan():
    """Test endpoint with hardcoded data"""
    if planner is None:
        raise HTTPException(status_code=503, detail="Planner not initialized")
    
    try:
        test_data = {
            "subjects": ["Mathematics", "Physics"],
            "subject_difficulty": ["Hard", "Medium"],
            "daily_hours_available": 3.0,
            "focus_duration": 45,
            "break_duration": 10,
            "stress_tolerance": 0.7,
            "goal": "Test API Endpoint",
            "deadline": "2024-12-31",
            "chronotype": "morning"
        }
        
        print("🧪 Running test plan generation...")
        daily_plans = planner.generate_study_plan(test_data)
        
        return {
            "success": True,
            "test_data": test_data,
            "plans_generated": len(daily_plans),
            "first_plan": daily_plans[0] if daily_plans else None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)