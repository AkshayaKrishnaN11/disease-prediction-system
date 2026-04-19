# ============================================
# Frontend Dockerfile — Next.js
# Multi-stage build for production
# ============================================
FROM node:20-alpine AS base

WORKDIR /app

# ── Stage 1: Install dependencies ──
FROM base AS deps
COPY package.json package-lock.json* ./
RUN npm ci --only=production 2>/dev/null || npm install

# ── Stage 2: Build ──
FROM base AS builder
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build argument for API URL (set at build time)
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# ── Stage 3: Production runner ──
FROM base AS runner
ENV NODE_ENV=production

# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
