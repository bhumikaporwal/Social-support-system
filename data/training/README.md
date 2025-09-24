# UAE Social Support System - LLM Training Data

## Overview
This directory contains training data for the UAE Social Support System's LLM chatbot. The data is designed to train the model on specific conversations, FAQs, and scenarios related to social support applications in the UAE.

## Current Data Sources

### 1. No Custom Training Data (Current State)
- **Model**: Ollama Llama2:7b-chat (pre-trained)
- **System**: Using general pre-trained model without domain-specific fine-tuning
- **Issue**: Generic responses not tailored to UAE social support context

### 2. Synthetic Application Data
- **File**: `../synthetic/generate_test_data.py`
- **Purpose**: Generates fake application data for testing
- **Content**: Emirates ID, bank statements, resumes, assets/liabilities
- **Note**: Not used for LLM training, only for application processing tests

## New Training Data Files

### 1. Comprehensive Training Dataset
**File**: `llm_training_data.json`

**Contents**:
- **10 Complete Conversations**: Real-world scenarios covering eligibility, documents, processing, troubleshooting
- **10 FAQ Entries**: Common questions with proper UAE social support answers
- **Context Examples**: How to handle different application states and user contexts
- **Training Scenarios**: Different user types (first-time, technical issues, borderline eligibility)
- **System Prompts**: Various prompt versions for different conversation styles

**Categories Covered**:
- Eligibility criteria
- Document requirements
- Application processing
- Program differences (financial vs economic enablement)
- Troubleshooting and appeals
- Family applications
- Technical support
- Financial assessment

### 2. Ollama Fine-tuning Format
**File**: `ollama_fine_tuning_format.jsonl`

**Format**: JSONL (JSON Lines) format compatible with Ollama fine-tuning
**Structure**: Each line contains a complete conversation with system, user, and assistant messages
**Usage**: Can be used to fine-tune Llama2 model for UAE social support domain

## Training Data Statistics

- **Total Conversations**: 10 complete multi-turn conversations
- **Total FAQ Entries**: 10 frequently asked questions
- **Categories**: 10 different topic areas
- **Languages**: English (Arabic support planned)
- **Context Examples**: 3 different application states
- **Training Scenarios**: 5 different user personas

## How This Data Addresses Previous Issues

### Problem: Same Responses (87%, 3500 AED)
**Root Causes**:
1. ❌ **Mock responses** in Streamlit app (hardcoded values)
2. ❌ **Low temperature** (0.1-0.3) making responses deterministic
3. ❌ **Generic system prompt** not specific to UAE context
4. ❌ **No domain training** - using general Llama2 model

**Solutions Applied**:
1. ✅ **Real LLM calls** - Replaced mock responses with actual chatbot
2. ✅ **Higher temperature** - Increased to 0.7-0.9 for variety
3. ✅ **Concise system prompt** - Shortened from 17 to 3 lines
4. ✅ **Random variations** - Added prompt variations and temperature ranges
5. ✅ **Training data** - Created UAE-specific conversation examples

## Usage Instructions

### For Fine-tuning Ollama Models

1. **Use the JSONL format**:
   ```bash
   ollama create social-support-uae -f Modelfile
   # Where Modelfile references the training data
   ```

2. **Training command example**:
   ```bash
   ollama train llama2:7b-chat ./data/training/ollama_fine_tuning_format.jsonl
   ```

### For Prompt Engineering

1. **Use conversation examples** from `llm_training_data.json`
2. **Implement few-shot learning** with conversation snippets
3. **Context injection** using the context examples

### For Testing and Validation

1. **FAQ Testing**: Use the 10 FAQ entries to test response accuracy
2. **Scenario Testing**: Test against the 5 different training scenarios
3. **Context Testing**: Verify proper handling of application states

## Data Quality Metrics

- **Conversation Length**: Average 3-4 turns per conversation
- **Response Length**: 50-200 words per assistant response
- **Topic Coverage**: 100% of main user journey covered
- **Tone Consistency**: Professional, empathetic, helpful throughout
- **UAE Specificity**: All monetary values, laws, and processes UAE-specific

## Future Enhancements

1. **Arabic Language Support**: Add Arabic conversations and responses
2. **More Scenarios**: Add edge cases and complex situations
3. **Multi-modal**: Include document analysis training examples
4. **User Feedback Integration**: Use real user conversations to improve data
5. **A/B Testing**: Create multiple response variations for testing

## Model Performance Expectations

After training on this data, the model should:
- ✅ Give varied responses instead of repeating "87%, 3500 AED"
- ✅ Provide UAE-specific eligibility information
- ✅ Handle document upload and technical issues appropriately
- ✅ Distinguish between financial support and economic enablement
- ✅ Show empathy for users in difficult financial situations
- ✅ Guide users through the complete application process

## Data Maintenance

- **Review Quarterly**: Update with new policies or program changes
- **User Feedback**: Incorporate common questions from real users
- **Performance Monitoring**: Track response quality and user satisfaction
- **Version Control**: Maintain different versions for A/B testing