from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
from groq import Groq

app = FastAPI(title="Tool Pricing Scraper API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PricingTier(BaseModel):
    tier_name: str
    monthly_price: float
    annual_price: float
    currency: str
    billing_cycle: str
    features_json: Dict[str, Any]
    limits_json: Dict[str, Any]

class ToolPricing(BaseModel):
    tool_id: int
    tool_name: str
    pricing_tiers: List[PricingTier]
    source_url: str

# Initialize Groq client
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return Groq(api_key=api_key)

async def fetch_tools():
    """Fetch tools from the Xano API"""
    url = "https://xwog-4ywl-hcgl.n7e.xano.io/api:Om3nin98/tool"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

async def search_pricing_page(tool_name: str, website_url: str = None):
    """Search for pricing page or construct from website URL"""
    
    # Method 1: Try website URL with common paths
    if website_url:
        from urllib.parse import urlparse, urljoin
        parsed = urlparse(website_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        pricing_paths = ['/pricing', '/pricing/', '/plans', '/plan', '/buy', '/purchase', '/subscribe']
        for path in pricing_paths:
            potential_url = urljoin(base_url, path)
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                    response = await client.head(potential_url, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    })
                    if response.status_code == 200:
                        print(f"Found pricing page: {potential_url}")
                        return potential_url
            except:
                continue
    
    # Method 2: Use Google search via ScraperAPI
    api_key = os.getenv("SCRAPERAPI_API_KEY")
    if api_key:
        try:
            query = f"{tool_name} pricing"
            google_url = f"https://www.google.com/search?q={query}"
            scraper_url = f"http://api.scraperapi.com?api_key={api_key}&url={google_url}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(scraper_url)
                if response.status_code == 200:
                    content = response.text
                    # Extract first result URL from Google search results
                    import re
                    urls = re.findall(r'href="(/url\?q=https?://[^&"]+)', content)
                    if urls:
                        result_url = urls[0].replace('/url?q=', '')
                        result_url = result_url.split('&')[0]
                        from urllib.parse import unquote
                        result_url = unquote(result_url)
                        if 'pricing' in result_url.lower() or 'plans' in result_url.lower():
                            return result_url
                        return result_url
        except Exception as e:
            print(f"Google search via ScraperAPI error: {e}")
    
    # Method 3: Fallback to constructed URL
    if website_url:
        return website_url.rstrip('/') + '/pricing'
    
    clean_name = tool_name.lower().replace(' ', '').replace('.', '')
    return f"https://www.{clean_name}.com/pricing"

async def scrape_with_scraperapi(url: str) -> Optional[str]:
    """Scrape using ScraperAPI (scraperapi.com) - FREE tier: 5,000 requests/month"""
    api_key = os.getenv("SCRAPERAPI_API_KEY")
    if not api_key:
        raise ValueError("SCRAPERAPI_API_KEY environment variable not set")
    
    if url.startswith('//'):
        url = 'https:' + url
    
    # Build ScraperAPI URL with parameters
    scraper_url = f"http://api.scraperapi.com?api_key={api_key}&url={url}&render=false"
    
    try:
        print(f"Scraping with ScraperAPI: {url}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(scraper_url)
            if response.status_code == 200:
                print(f"Successfully scraped (length: {len(response.text)} chars)")
                return response.text[:15000]
            else:
                print(f"ScraperAPI returned status code: {response.status_code}")
                return None
    except Exception as e:
        print(f"ScraperAPI error: {e}")
        return None

def extract_pricing_with_groq(tool_name: str, scraped_content: str):
    """Use Groq to extract structured pricing information"""
    client = get_groq_client()
    
    prompt = f"""Extract pricing information for {tool_name} from the following HTML/text content.
Return a JSON array of pricing tiers with this exact structure:
[
  {{
    "tier_name": "Free/Starter/Pro/Enterprise",
    "monthly_price": 0,
    "annual_price": 0,
    "currency": "USD",
    "billing_cycle": "monthly/annual/one-time",
    "features_json": {{
      "records": "number or 'Unlimited'",
      "storage": "e.g., '1GB', '10GB'",
      "features": ["feature 1", "feature 2"],
      "api_requests": "number or 'Unlimited'"
    }},
    "limits_json": {{
      "records": "number or -1 for unlimited",
      "storage_gb": "number or -1 for unlimited",
      "api_requests": "number or -1 for unlimited"
    }}
  }}
]

IMPORTANT:
- Extract ALL pricing tiers you can find
- If features/limits aren't clear, use reasonable defaults
- For free tiers, use 0 for prices
- For unlimited, use -1 in limits_json
- Include as many relevant features as you can find
- If the tool is open-source/free with no paid plans, return a single "Free" tier

Content:
{scraped_content[:10000]}

Return ONLY valid JSON array, no additional text."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a pricing data extraction expert. Return only valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=3000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
            result = result.strip()
        
        parsed = json.loads(result)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception as e:
        print(f"Groq extraction error: {e}")
        return []

async def upload_pricing_to_xano(tool_id: int, pricing_tiers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Upload pricing tiers to Xano API"""
    url = "https://xwog-4ywl-hcgl.n7e.xano.io/api:Om3nin98/pricing_tier"
    results = []
    
    import time
    current_timestamp = int(time.time() * 1000)  # Current time in milliseconds
    
    async with httpx.AsyncClient() as client:
        for tier in pricing_tiers:
            # Transform to Xano format
            xano_data = {
                "tool": tool_id,
                "tier_name": tier.get("tier_name", ""),
                "monthly_price": tier.get("monthly_price", 0),
                "annual_price": tier.get("annual_price", 0),
                "currency": tier.get("currency", "USD"),
                "billing_cycle": tier.get("billing_cycle", "monthly"),
                "features_json": tier.get("features_json", {}),
                "limits_json": tier.get("limits_json", {}),
                "is_current": True,
                "effective_from": current_timestamp,
                "effective_to": 0,
                "updated_at": current_timestamp
            }
            
            try:
                response = await client.post(url, json=xano_data)
                if response.status_code == 200 or response.status_code == 201:
                    result_data = response.json()
                    results.append({
                        "status": "success",
                        "tier_name": tier.get("tier_name"),
                        "xano_response": result_data
                    })
                    print(f"  ✓ Uploaded tier: {tier.get('tier_name')}")
                else:
                    results.append({
                        "status": "failed",
                        "tier_name": tier.get("tier_name"),
                        "error": f"HTTP {response.status_code}",
                        "response": response.text
                    })
                    print(f"  ✗ Failed to upload tier: {tier.get('tier_name')} (HTTP {response.status_code})")
            except Exception as e:
                results.append({
                    "status": "error",
                    "tier_name": tier.get("tier_name"),
                    "error": str(e)
                })
                print(f"  ✗ Error uploading tier: {tier.get('tier_name')} - {e}")
    
    return results

@app.get("/")
async def root():
    return {
        "message": "Tool Pricing Scraper API with ScraperAPI",
        "endpoints": [
            "/tools",
            "/pricing/{tool_id}?upload=true",
            "/pricing/all?limit=10&upload=true",
            "/config"
        ],
        "scraping_service": "ScraperAPI (5,000 free requests/month)",
        "usage": {
            "get_single": "GET /pricing/1 - Extract pricing for tool ID 1",
            "upload_single": "GET /pricing/1?upload=true - Extract and upload to Xano",
            "get_all": "GET /pricing/all?limit=5 - Extract pricing for 5 tools",
            "upload_all": "GET /pricing/all?limit=5&upload=true - Extract and upload all to Xano"
        }
    }

@app.get("/config")
async def get_config():
    """Check which API keys are configured"""
    return {
        "groq_api": bool(os.getenv("GROQ_API_KEY")),
        "scraperapi": bool(os.getenv("SCRAPERAPI_API_KEY"))
    }

@app.get("/tools")
async def get_tools():
    """Get all tools from Xano API"""
    try:
        tools = await fetch_tools()
        return {"count": len(tools), "tools": tools}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pricing/{tool_id}")
async def get_tool_pricing(tool_id: int, upload: bool = False):
    """Get pricing for a specific tool and optionally upload to Xano
    
    Parameters:
    - tool_id: The ID of the tool
    - upload: If True, automatically upload extracted pricing to Xano API
    """
    try:
        tools = await fetch_tools()
        tool = next((t for t in tools if t['id'] == tool_id), None)
        
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        tool_name = tool['name']
        website_url = tool.get('website_url')
        
        pricing_url = await search_pricing_page(tool_name, website_url)
        print(f"Found pricing URL: {pricing_url}")
        
        content = await scrape_with_scraperapi(pricing_url)
        
        if not content:
            return {
                "tool_id": tool_id,
                "tool_name": tool_name,
                "error": "Could not scrape pricing page",
                "attempted_url": pricing_url,
                "suggestion": "Make sure SCRAPERAPI_API_KEY is configured"
            }
        
        pricing_tiers = extract_pricing_with_groq(tool_name, content)
        
        response_data = {
            "tool_id": tool_id,
            "tool_name": tool_name,
            "pricing_tiers": pricing_tiers,
            "source_url": pricing_url
        }
        
        # Upload to Xano if requested
        if upload and pricing_tiers:
            print(f"\nUploading {len(pricing_tiers)} pricing tier(s) to Xano...")
            upload_results = await upload_pricing_to_xano(tool_id, pricing_tiers)
            response_data["upload_results"] = upload_results
            response_data["uploaded"] = True
        
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pricing/all")
async def get_all_pricing(limit: int = 10, upload: bool = False):
    """Get pricing for all tools and optionally upload to Xano
    
    Parameters:
    - limit: Maximum number of tools to process
    - upload: If True, automatically upload all extracted pricing to Xano API
    """
    try:
        tools = await fetch_tools()
        results = []
        
        for tool in tools[:limit]:
            try:
                tool_id = tool['id']
                tool_name = tool['name']
                website_url = tool.get('website_url')
                
                print(f"\n{'='*50}")
                print(f"Processing {tool_name}...")
                print(f"{'='*50}")
                
                pricing_url = await search_pricing_page(tool_name, website_url)
                content = await scrape_with_scraperapi(pricing_url)
                
                if content:
                    pricing_tiers = extract_pricing_with_groq(tool_name, content)
                    
                    result_entry = {
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "pricing_tiers": pricing_tiers,
                        "source_url": pricing_url,
                        "status": "success"
                    }
                    
                    # Upload to Xano if requested
                    if upload and pricing_tiers:
                        print(f"\nUploading {len(pricing_tiers)} pricing tier(s) to Xano...")
                        upload_results = await upload_pricing_to_xano(tool_id, pricing_tiers)
                        result_entry["upload_results"] = upload_results
                        result_entry["uploaded"] = True
                        
                        # Count successful uploads
                        successful_uploads = sum(1 for r in upload_results if r.get("status") == "success")
                        print(f"✓ Successfully uploaded {successful_uploads}/{len(pricing_tiers)} tier(s)")
                    
                    results.append(result_entry)
                    print(f"✓ Successfully extracted {len(pricing_tiers)} pricing tier(s)")
                else:
                    results.append({
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "error": "Could not scrape pricing",
                        "source_url": pricing_url,
                        "status": "failed"
                    })
                    print(f"✗ Failed to scrape")
            except Exception as e:
                print(f"✗ Error processing {tool.get('name', 'unknown')}: {e}")
                results.append({
                    "tool_id": tool.get('id'),
                    "tool_name": tool.get('name', 'unknown'),
                    "error": str(e),
                    "status": "error"
                })
        
        success_count = sum(1 for r in results if r.get("status") == "success")
        uploaded_count = sum(1 for r in results if r.get("uploaded"))
        
        return {
            "total": len(results),
            "success": success_count,
            "failed": len(results) - success_count,
            "uploaded": uploaded_count if upload else 0,
            "pricing_data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
