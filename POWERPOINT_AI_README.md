# PowerPoint AI Assistant

A Python program that uses AI to generate and enhance PowerPoint presentations, similar to Fastlane for PowerPoint.

## Features

- **Generate New Presentations**: Create complete presentations from a topic using AI
- **Enhance Existing Presentations**: Improve slide content with AI-powered suggestions
- **Professional Formatting**: Automatically formats slides with proper sizing and styling
- **Speaker Notes**: Generates helpful speaker notes for each slide

## Installation

1. Ensure you have Python 3.8+ installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Generate a New Presentation

```bash
python powerpoint_ai.py generate "Topic Name" -n 5 -o output.pptx
```

Parameters:
- `topic`: The topic for your presentation (required)
- `-n, --num-slides`: Number of slides to generate (default: 5)
- `-o, --output`: Output file name (default: presentation.pptx)

Example:
```bash
python powerpoint_ai.py generate "Digital Marketing Strategy" -n 7 -o marketing.pptx
```

### Enhance an Existing Presentation

```bash
python powerpoint_ai.py enhance input.pptx -o enhanced.pptx
```

Parameters:
- `input`: Path to existing PowerPoint file (required)
- `-o, --output`: Output file name (default: input_enhanced.pptx)

Example:
```bash
python powerpoint_ai.py enhance old_presentation.pptx -o improved.pptx
```

## How It Works

1. **Generate Mode**: 
   - Takes your topic and desired number of slides
   - Uses Gemini AI to create a structured outline
   - Builds a professional PowerPoint with title, content slides, and speaker notes

2. **Enhance Mode**:
   - Reads your existing PowerPoint file
   - Analyzes each slide's content
   - Uses AI to improve titles and bullet points
   - Saves enhanced version while preserving formatting

## API Integration

Uses Google's Gemini AI (gemini-2.0-flash-exp model) for:
- Content generation
- Professional wording improvements
- Structured presentation outlines

## Dependencies

- `python-pptx`: PowerPoint file manipulation
- `google-generativeai`: Gemini AI integration
- `python-dotenv`: Environment variable management

See `requirements.txt` for full list.

## License

MIT License - Free to use and modify
