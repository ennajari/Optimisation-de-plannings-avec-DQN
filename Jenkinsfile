pipeline {
    agent any

    environment {
        DOCKER_HUB = credentials('docker-hub-credentials')
    }

    stages {
        stage('Build') {
            steps {
                sh 'docker build -t time-management-assistant .'
            }
        }

        stage('Test') {
            steps {
                sh 'docker run time-management-assistant python -m pytest tests/'
            }
        }

        stage('Push to Docker Hub') {
            steps {
                sh 'docker login -u $DOCKER_HUB_USR -p $DOCKER_HUB_PSW'
                sh 'docker tag time-management-assistant yourusername/time-management-assistant:latest'
                sh 'docker push yourusername/time-management-assistant:latest'
            }
        }

        stage('Deploy') {
            steps {
                sh 'docker-compose down'
                sh 'docker-compose up -d'
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}