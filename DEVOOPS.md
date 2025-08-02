# 📦 py-fmg Deployment Guide

This guide helps you deploy the `py-fmg` application using either **Docker Swarm** or **Kubernetes**, with:

* ✅ Scalable app service (FastAPI)
* 🗺️ PostGIS-enabled PostgreSQL database
* 🌐 NGINX reverse proxy
* ⚖️ Load balancing
* 🔄 Auto-restart & replication

---

## 📁 Folder Structure

```bash
.
├── docker-compose.yml
├── docker-compose.prod.yml
├── nginx.conf
├── k8s/
│   ├── py-fmg-deployment.yaml
│   ├── postgis-db.yaml
│   └── nginx.yaml
└── .env
```

---

## 🚀 Option 1: Docker Swarm Deployment

### ✅ Step 1: Initialize Swarm

```bash
docker swarm init
```

> If deploying on multiple nodes, copy and run the token from this command on worker nodes.

---

### ✅ Step 2: Deploy Stack

```bash
docker stack deploy -c docker-compose.yml pyfmg
```

### 📌 Notes:

* Set environment variables in `.env`:

  ```env
  DB_USER=user
  DB_PASSWORD=password
  DB_NAME=py-fmg
  DB_PORT=5432
  ```

* Docker Swarm will automatically:

  * Create a service for the app with multiple replicas
  * Start the PostGIS container
  * Load balance via NGINX (if added as a service)

---

## ☸️ Option 2: Kubernetes Deployment

### ✅ Step 1: Create Namespace (Optional)

```bash
kubectl create namespace pyfmg
```

---

### ✅ Step 2: Apply Resources

```bash
kubectl apply -f k8s/
```

> This includes:
>
> * FastAPI app deployment and service
> * PostGIS DB deployment, secret, PVC, and service
> * NGINX reverse proxy deployment, config map, and LoadBalancer service

---

### ✅ Step 3: Access the App

* If using **minikube**:

```bash
minikube service nginx-lb
```

* On cloud providers (GKE, EKS, AKS), use the **external IP** from:

```bash
kubectl get svc nginx-lb
```

---

## ⚙️ Scaling

### 🚀 Docker Swarm

```bash
docker service scale pyfmg_web=5
```

### ☸️ Kubernetes

```bash
kubectl scale deployment py-fmg-app --replicas=5
```

---

## 🧪 Testing

After deployment, test the endpoint:

```bash
curl http://<nginx-ip>/maps/generate
```

Check logs with:

```bash
# Docker Swarm
docker service logs pyfmg_web

# Kubernetes
kubectl logs -l app=py-fmg
```

---

## 📘 Future Improvements

* Add Ingress + TLS (e.g. cert-manager)
* Add Horizontal Pod Autoscaler (HPA) in Kubernetes
* Add health/readiness/liveness probes
* Migrate config to Helm chart for easier management

