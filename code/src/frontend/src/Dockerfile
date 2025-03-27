FROM node:16-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build # Replace with your build command

FROM nginx:alpine

COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80
