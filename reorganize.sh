#!/bin/bash

# AI Agents Portfolio - Repository Reorganization Script
# This script will reorganize your current structure to the new organized format

echo "🚀 Starting AI Agents Portfolio Reorganization..."

# Create the main project structure
echo "📁 Creating main directory structure..."

# Create directories for all 10 agents
declare -a agents=(
    "01-stock-price-prediction"
    "02-multimodal-sentiment-analysis" 
    "03-advanced-recommendation-engine"
    "04-ai-driven-customer-segmentation"
    "05-realtime-fraud-detection"
    "06-predictive-healthcare-analytics"
    "07-realtime-autonomous-image-recognition"
    "08-genai-smart-retail-experience"
    "09-genai-customer-support"
    "10-predictive-maintenance-systems"
)

# Create directory structure for each agent
for agent in "${agents[@]}"; do
    echo "Creating structure for $agent..."
    mkdir -p "$agent/code-implementation"
    mkdir -p "$agent/nocode-implementation"
    
    # Create basic files for each implementation
    touch "$agent/README.md"
    touch "$agent/code-implementation/main.py"
    touch "$agent/code-implementation/requirements.txt"
    touch "$agent/code-implementation/.env.example"
    touch "$agent/nocode-implementation/workflow.json"
    touch "$agent/nocode-implementation/README.md"
done

# Move existing files from old structure to new structure
echo "📦 Moving existing files..."

# Check if old structure exists and move files
if [ -d "09-genai-customer-support" ]; then
    echo "Moving files from old 09-genai-customer-support structure..."
    
    # Move existing files to code-implementation
    if [ -f "09-genai-customer-support/main.py" ]; then
        mv 09-genai-customer-support/main.py 09-genai-customer-support/code-implementation/
    fi
    
    if [ -f "09-genai-customer-support/requirements.txt" ]; then
        mv 09-genai-customer-support/requirements.txt 09-genai-customer-support/code-implementation/
    fi
    
    # Move any other files
    for file in 09-genai-customer-support/*; do
        if [ -f "$file" ] && [ "$(basename "$file")" != "README.md" ]; then
            mv "$file" 09-genai-customer-support/code-implementation/
        fi
    done
    
    echo "✅ Existing files moved successfully"
else
    echo "ℹ️  No existing 09-genai-customer-support directory found"
fi

echo "🎉 Repository reorganization complete!"
echo ""
echo "📋 Next steps:"
echo "1. Review the new structure"
echo "2. Update your existing code files"
echo "3. Add your n8n workflows to nocode-implementation folders"
echo "4. Update README files for each agent"