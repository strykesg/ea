# Deploying to Dokploy

This guide will help you deploy the AI Training Data Generator to Dokploy.

## Prerequisites

- A Dokploy instance running on your VPS
- Git repository with your code (GitHub, GitLab, or Bitbucket)
- Your OpenRouter API key

## Deployment Methods

### Method 1: Dockerfile (Recommended for Production)

This method uses the included `Dockerfile` for a rock-solid production deployment.

#### Step 1: Push Code to Git Repository

Make sure your code is pushed to a Git repository (GitHub/GitLab/Bitbucket).

#### Step 2: Create Application in Dokploy

1. Log into your Dokploy dashboard (usually at `http://your-vps-ip:3000`)
2. Click **"+ Create Service"** → **"Application"**
3. Give your app a name (e.g., "ai-data-generator")

#### Step 3: Connect Git Repository

1. Select your Git provider (GitHub/GitLab/Bitbucket)
2. Authorize Dokploy to access your repositories
3. Select the repository containing this code
4. Choose the branch (usually `main` or `master`)

#### Step 4: Configure Build Settings

1. **Build Type**: Select **"Dockerfile"**
2. **Dockerfile Path**: Leave as `/Dockerfile` (default)
3. **Build Context**: Leave as `/` (default)

#### Step 5: Set Environment Variables

Go to the **"Environment"** tab and add these variables:

**Required:**
```
OPENROUTER_API_KEY=your_api_key_here
```

**Optional (customize as needed):**
```
GENERATOR_MODEL=mistralai/mistral-nemo
REFINER_MODEL=deepseek/deepseek-r1-0528-qwen3-8b
MAX_WORKERS=10
MAX_CONCURRENT_REQUESTS=20
SESSION_SECRET=your_random_secret_here
```

#### Step 6: Configure Port

1. Go to **"Settings"** → **"Ports"**
2. Set **Container Port**: `5000`
3. Set **Published Port**: `5000` (or your preferred port)

#### Step 7: Deploy

1. Click the **"Deploy"** button
2. Monitor the build logs in real-time
3. Wait for deployment to complete (usually 2-5 minutes)

#### Step 8: Add Domain

1. Go to the **"Domains"** tab
2. Either:
   - Generate a free domain with traefik.me, OR
   - Add your custom domain
3. Set the port to `5000`
4. Enable **"SSL/HTTPS"** (automatic with Let's Encrypt)

#### Step 9: Access Your Application

Your app will be available at:
- Free domain: `http://your-app.traefik.me`
- Custom domain: `https://yourdomain.com`

---

### Method 2: Docker Compose (For Complex Setups)

If you need additional services (future expansion), use Docker Compose:

1. In Dokploy, click **"+ Create Service"** → **"Compose"**
2. Name your service
3. Paste the contents of `docker-compose.yml`
4. Add environment variables in the Dokploy UI
5. Click **"Deploy"**

---

## Production Best Practices

### 1. **Use GitHub Actions for CI/CD**

For better performance, build images in CI/CD instead of on your VPS:

Create `.github/workflows/deploy.yml`:

```yaml
name: Build and Deploy to Dokploy

on:
  push:
    branches: [main]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build and push Docker image
        run: |
          docker build -t yourusername/ai-data-generator:latest .
          echo ${{ secrets.DOCKERHUB_TOKEN }} | docker login -u ${{ secrets.DOCKERHUB_USERNAME }} --password-stdin
          docker push yourusername/ai-data-generator:latest
      
      - name: Trigger Dokploy deployment
        run: |
          curl -X POST ${{ secrets.DOKPLOY_WEBHOOK_URL }}
```

Then in Dokploy:
1. Change build type to **"Docker Image"**
2. Enter image: `yourusername/ai-data-generator:latest`
3. Enable webhook auto-deploy

### 2. **Volume Persistence**

To persist generated data across deployments:

1. Go to **"Volumes"** tab in Dokploy
2. Add volume mount:
   - **Container Path**: `/app/output`
   - **Host Path**: `/data/ai-generator/output`

### 3. **Resource Limits**

Set appropriate resource limits:

1. Go to **"Advanced"** → **"Resources"**
2. Set memory limit: `2GB` (minimum recommended)
3. Set CPU limit: `2 cores` (for 10 workers)

Scale based on your `MAX_WORKERS` setting:
- 5 workers: 1GB RAM, 1 CPU
- 10 workers: 2GB RAM, 2 CPUs
- 20 workers: 4GB RAM, 4 CPUs

### 4. **Monitoring**

Monitor your application:
- **Logs**: View real-time logs in Dokploy **"Logs"** tab
- **Metrics**: Check CPU/Memory usage in **"Monitoring"** tab
- **Health**: Application health check runs every 30 seconds

### 5. **Scaling**

To handle more load:
1. Increase `MAX_WORKERS` environment variable
2. Increase resource limits accordingly
3. Redeploy the application

---

## Troubleshooting

### Build Fails

**Issue**: Docker build fails
**Solution**: Check build logs in Dokploy. Common issues:
- Missing dependencies → Check `pyproject.toml`
- Out of memory → Increase VPS RAM or use CI/CD builds

### App Doesn't Start

**Issue**: Container starts but app crashes
**Solution**: 
1. Check application logs in Dokploy
2. Verify `OPENROUTER_API_KEY` is set correctly
3. Ensure port 5000 is correctly configured

### Can't Access Dashboard

**Issue**: Domain doesn't load
**Solution**:
1. Verify domain is correctly configured
2. Check SSL certificate status
3. Ensure firewall allows ports 80/443
4. Check that port 5000 is set in domain settings

### High Memory Usage

**Issue**: Container uses too much RAM
**Solution**:
1. Reduce `MAX_WORKERS` (try 5 instead of 10)
2. Reduce `MAX_CONCURRENT_REQUESTS` (try 10 instead of 20)
3. Increase VPS RAM if needed

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | Your OpenRouter API key |
| `GENERATOR_MODEL` | No | `mistralai/mistral-nemo` | Fast LLM for generation |
| `REFINER_MODEL` | No | `deepseek/deepseek-r1-0528-qwen3-8b` | Powerful LLM for refinement |
| `MAX_WORKERS` | No | `10` | Number of concurrent workers |
| `MAX_CONCURRENT_REQUESTS` | No | `20` | API request concurrency limit |
| `SESSION_SECRET` | No | auto-generated | Secret key for sessions |

---

## Local Testing

Before deploying to Dokploy, test locally with Docker:

```bash
# Build the image
docker build -t ai-data-generator .

# Run with docker-compose
docker-compose up

# Access at http://localhost:5000
```

---

## Support

- **Dokploy Docs**: https://docs.dokploy.com
- **Dokploy GitHub**: https://github.com/Dokploy/dokploy
- **Issues**: Report in your repository's issue tracker

---

## Quick Checklist

- [ ] Code pushed to Git repository
- [ ] Dokploy instance installed and running
- [ ] Application created in Dokploy
- [ ] Git repository connected
- [ ] Dockerfile build type selected
- [ ] Environment variables added (especially `OPENROUTER_API_KEY`)
- [ ] Port configured (5000)
- [ ] Application deployed successfully
- [ ] Domain configured with SSL
- [ ] Health check passing
- [ ] Dashboard accessible and working

---

**🎉 Once deployed, your AI Training Data Generator will be live and ready to generate high-quality training datasets!**
