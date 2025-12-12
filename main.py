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
    description="Generate personalized study plans using pre-trained ECLAT model",
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

# Global planner instance - loads pre-trained model on startup
planner = None

@app.on_event("startup")
async def startup_event():
    """Initialize on startup - LOADS PRE-TRAINED MODEL"""
    global planner
    try:
        print("🚀 Starting Study Planner...")
        planner = StudyPlanner()  # This loads the pre-trained model
        print("✅ Study Planner initialized successfully!")
        
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
        "status": "ready" if planner else "initializing",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "POST /generate-plan": "Generate study plan",
            "GET /rules": "Get ECLAT association rules",
            "GET /status": "Get system status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if planner is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": planner is not None and planner.rules is not None
    }

@app.get("/status")
async def get_status():
    """Get system status"""
    if planner is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    status = {
        "service": "ready",
        "model_loaded": planner.rules is not None,
        "rules_count": len(planner.rules) if planner.rules is not None else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    return status

@app.get("/rules")
async def get_rules(limit: int = 10):
    """Get ECLAT association rules"""
    if planner is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
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
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        print(f"📥 Generating plan for: {request.goal}")
        
        # Convert request to dict
        user_input = request.dict()
        user_input['deadline'] = str(user_input['deadline'])
        
        # Generate study plan
        daily_plans = planner.generate_study_plan(user_input)
        
        if not daily_plans:
            raise HTTPException(status_code=400, detail="Could not generate study plan")
        
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
        
        print(f"✅ Plan generated: {len(daily_plans)} days, {total_hours:.1f} hours")
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"🔥 Error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating plan: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
