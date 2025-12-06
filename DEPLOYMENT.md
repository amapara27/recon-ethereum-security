# Deployment Guide

## Issue: Vercel Deployment Not Connecting to AWS Backend

### Root Cause
The deployment fails to connect to the AWS backend due to **Mixed Content Policy**. Vercel serves your frontend over HTTPS, but your AWS backend runs on HTTP. Modern browsers block HTTPS pages from making HTTP requests for security reasons.

### Solution Options

#### Option 1: Enable HTTPS on AWS Backend (Recommended)

You need to configure HTTPS on your AWS EC2 instance. Here are the steps:

1. **Install Nginx and Certbot on your EC2 instance:**
   ```bash
   ssh into your EC2 instance
   sudo apt update
   sudo apt install nginx certbot python3-certbot-nginx
   ```

2. **Configure Nginx as a reverse proxy:**
   Create `/etc/nginx/sites-available/fraud-detector`:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;  # Or use your EC2 public IP

       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Get SSL certificate (if you have a domain):**
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```

4. **Update Vercel Environment Variable:**
   - Go to Vercel Dashboard → Your Project → Settings → Environment Variables
   - Add: `VITE_API_URL` = `https://your-domain.com` (or `https://your-ec2-ip`)

#### Option 2: Use Vercel Serverless Functions as Proxy (Alternative)

Create a Vercel serverless function that proxies requests to your AWS backend:

1. Create `api/get-alerts.js` in your project root:
   ```javascript
   export default async function handler(req, res) {
     const response = await fetch('http://54.186.52.217:8000/api/get-alerts');
     const data = await response.json();
     res.status(200).json(data);
   }
   ```

2. Update frontend to use `/api/get-alerts` instead

#### Option 3: Temporary - Allow Insecure Content (NOT Recommended for Production)

This is only for testing and should NOT be used in production:
- Chrome: Click the shield icon in the address bar → "Load unsafe scripts"
- This defeats the purpose of HTTPS and exposes users to security risks

### Current Setup

The project is now configured to use environment variables:
- **Local Development**: Uses `frontend/.env.local` → `http://54.186.52.217:8000`
- **Production**: Uses `frontend/.env.production` → `http://54.186.52.217:8000`

### Vercel Deployment Steps

1. **Set Environment Variable in Vercel:**
   - Go to: Vercel Dashboard → Your Project → Settings → Environment Variables
   - Add variable:
     - Key: `VITE_API_URL`
     - Value: `https://your-backend-url.com` (once HTTPS is configured)
     - Environment: Production

2. **Redeploy:**
   ```bash
   git add .
   git commit -m "Fix API URL configuration"
   git push
   ```

3. **Verify:**
   - Check browser console for errors
   - Look for "Mixed Content" warnings
   - Verify API calls in Network tab

### Testing Locally

```bash
cd frontend
npm run dev
```

The app should connect to `http://54.186.52.217:8000` locally.

### Current Files Changed

- `frontend/src/App.jsx` - Now uses `import.meta.env.VITE_API_URL`
- `frontend/src/api.js` - Now uses `import.meta.env.VITE_API_URL`
- `frontend/.env.local` - Local development config
- `frontend/.env.production` - Production config
- `vercel.json` - Updated build configuration
