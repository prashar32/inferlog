# Build the SPA, then serve it as static files behind nginx (which also
# reverse-proxies the API services — see nginx.conf).
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
# The nginx image runs envsubst on files in /etc/nginx/templates at
# startup and writes the result into /etc/nginx/conf.d. We restrict
# substitution to a known env var so nginx-native $variables aren't
# mangled (see NGINX_ENVSUBST_FILTER in docker-compose.yml).
COPY default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
