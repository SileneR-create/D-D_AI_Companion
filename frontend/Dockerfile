# --- Frontend React : build Vite puis service statique via nginx ---
# Base glibc (pas alpine/musl) pour eviter le bug npm des deps natives de rollup.
FROM node:20-slim AS build
WORKDIR /app
# On ne copie QUE package.json (pas le lock potentiellement genere sous un autre OS)
# afin que npm resolve les binaires natifs adaptes a l'image Linux.
COPY package.json ./
RUN npm install --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
