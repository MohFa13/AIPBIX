# Power BI Chatbot Visual

A responsive chatbot custom visual for Power BI that allows users to interact with a chatbot endpoint directly within Power BI reports.

## Overview

This project contains a Power BI custom visual built with TypeScript that provides a chatbot interface. Users can configure an endpoint URL and chat with an AI service directly in Power BI.

## Prerequisites

- Node.js (v16 or higher)
- npm (comes with Node.js)
- Power BI Desktop (for testing the visual locally)

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```
   This will:
   - Compile the TypeScript code
   - Start a local development server
   - Provide a URL to load the visual in Power BI Desktop

3. Package the visual for distribution:
   ```bash
   npm run package
   ```
   This creates a `.pbiviz` file in the `dist/` directory that can be imported into Power BI.

## Development

### Project Structure

```
.
├── src/
│   ├── visual.ts       # Main visual implementation
│   └── settings.ts    # Visual settings and formatting model
├── assets/            # Visual assets (icon, etc.)
├── style/             # Visual styles
├── capabilities.json  # Visual capabilities definition
├── pbiviz.json        # Power BI visual configuration
├── package.json       # npm dependencies and scripts
├── tsconfig.json      # TypeScript configuration
└── streamlit_app.py   # Streamlit web interface (separate component)
```

### Available Scripts

- `npm start` - Start development server
- `npm run package` - Package the visual for distribution
- `npm run test` - Verify the visual package
- `npm run build` - Compile TypeScript code
- `npm run lint` - Run ESLint

### Configuration

The visual can be configured in Power BI Desktop:

1. Add the visual to a report
2. Open the Format pane
3. Under "Chatbot" → "Endpoint", enter your chatbot endpoint URL (e.g., `http://localhost:8000/chat`)

The chatbot will send POST requests to the configured endpoint with the following JSON payload:

```json
{
  "message": "user's message here"
}
```

And expects a response. The visual will attempt to extract text from the response by checking these fields in order:
- `response`
- `message`
- `text`
- Full JSON response if none of the above are found

## Streamlit Integration

This project also includes a Streamlit web interface (`streamlit_app.py`) that can be deployed to Streamlit Cloud. See `README_HOTFIX.md` for more details on the Streamlit setup.

## Troubleshooting

### `npm install` fails

Make sure you have Node.js installed:
```bash
node --version
npm --version
```

### Visual doesn't load in Power BI Desktop

1. Ensure the development server is running (`npm start`)
2. Copy the provided URL and load it in Power BI Desktop's developer mode
3. Check the console for any errors

### Chatbot endpoint not responding

1. Verify the endpoint URL is correct
2. Check that the endpoint accepts POST requests with JSON payloads
3. Ensure CORS is configured to allow requests from Power BI

## License

MIT

## Support

For issues and questions, please visit: https://example.com/support
