#!/bin/bash

# Django REST API Project Setup Script
# This script creates the complete project structure and all necessary files

echo "Creating Django REST API Project Structure..."

# Create main project directory
mkdir -p task-management-api
cd task-management-api

# Create project structure
mkdir -p taskmanager/api/migrations

# Create __init__.py files
touch taskmanager/__init__.py
touch api/__init__.py
touch api/migrations/__init__.py

# Create manage.py
cat > manage.py << 'EOF'
#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'taskmanager.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
Django==4.2.7
djangorestframework==3.14.0
pytz==2023.3
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
.venv

# Django
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal
media/
staticfiles/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment variables
.env
.env.local

# pytest
.pytest_cache/
.coverage
htmlcov/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json
EOF

# Create README.md
cat > README.md << 'EOF'
# Task Management REST API

A complete REST API built with Django and Django REST Framework for managing tasks and projects.

## Features

- Create, Read, Update, and Delete (CRUD) operations for tasks
- RESTful API endpoints
- SQLite database integration
- Data serialization with Django REST Framework
- Clean and organized project structure

## Technologies Used

- **Django 4.2+**: Web framework
- **Django REST Framework**: API development
- **SQLite**: Database
- **Python 3.8+**: Programming language

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/task-management-api.git
cd task-management-api
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

5. Create a superuser (optional):
```bash
python manage.py createsuperuser
```

6. Run the development server:
```bash
python manage.py runserver
```

The API will be available at `http://127.0.0.1:8000/`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/` | API overview with available endpoints |
| GET | `/api/tasks/` | Get all tasks |
| GET | `/api/tasks/<id>/` | Get a specific task |
| POST | `/api/tasks/create/` | Create a new task |
| PUT | `/api/tasks/<id>/update/` | Update a specific task |
| DELETE | `/api/tasks/<id>/delete/` | Delete a specific task |

## API Usage Examples

### Get All Tasks
```bash
curl http://127.0.0.1:8000/api/tasks/
```

### Create a Task
```bash
curl -X POST http://127.0.0.1:8000/api/tasks/create/ \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Complete project documentation",
    "description": "Write comprehensive README and API docs",
    "completed": false,
    "priority": "high"
  }'
```

### Update a Task
```bash
curl -X PUT http://127.0.0.1:8000/api/tasks/1/update/ \
  -H "Content-Type: application/json" \
  -d '{
    "completed": true
  }'
```

### Delete a Task
```bash
curl -X DELETE http://127.0.0.1:8000/api/tasks/1/delete/
```

## Model Schema

### Task Model

| Field | Type | Description |
|-------|------|-------------|
| id | Integer | Auto-generated primary key |
| title | String (200) | Task title |
| description | Text | Detailed task description |
| completed | Boolean | Task completion status |
| priority | String (20) | Priority level (low, medium, high) |
| created_at | DateTime | Timestamp of creation |
| updated_at | DateTime | Timestamp of last update |

## Running Tests
```bash
python manage.py test
```

## License

This project is licensed under the MIT License.
EOF

# Create taskmanager/settings.py
cat > taskmanager/settings.py << 'EOF'
"""
Django settings for taskmanager project.
"""

from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-your-secret-key-here-change-in-production'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'api',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'taskmanager.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'taskmanager.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ],
}
EOF

# Create taskmanager/urls.py
cat > taskmanager/urls.py << 'EOF'
"""
URL configuration for taskmanager project.
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
]
EOF

# Create taskmanager/asgi.py
cat > taskmanager/asgi.py << 'EOF'
"""
ASGI config for taskmanager project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'taskmanager.settings')

application = get_asgi_application()
EOF

# Create taskmanager/wsgi.py
cat > taskmanager/wsgi.py << 'EOF'
"""
WSGI config for taskmanager project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'taskmanager.settings')

application = get_wsgi_application()
EOF

# Create api/models.py
cat > api/models.py << 'EOF'
from django.db import models

class Task(models.Model):
    """
    Task model for managing tasks in the system.
    """
    PRIORITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]
    
    title = models.CharField(max_length=200, help_text="Title of the task")
    description = models.TextField(blank=True, help_text="Detailed description of the task")
    completed = models.BooleanField(default=False, help_text="Task completion status")
    priority = models.CharField(
        max_length=20,
        choices=PRIORITY_CHOICES,
        default='medium',
        help_text="Priority level of the task"
    )
    created_at = models.DateTimeField(auto_now_add=True, help_text="Timestamp when task was created")
    updated_at = models.DateTimeField(auto_now=True, help_text="Timestamp when task was last updated")
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Task'
        verbose_name_plural = 'Tasks'
    
    def __str__(self):
        return self.title
    
    def save(self, *args, **kwargs):
        """Override save method to add custom logic if needed"""
        super().save(*args, **kwargs)
EOF

# Create api/serializers.py
cat > api/serializers.py << 'EOF'
from rest_framework import serializers
from .models import Task

class TaskSerializer(serializers.ModelSerializer):
    """
    Serializer for Task model to convert complex data types
    into native Python data types that can be easily rendered into JSON.
    """
    
    class Meta:
        model = Task
        fields = [
            'id',
            'title',
            'description',
            'completed',
            'priority',
            'created_at',
            'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def validate_title(self, value):
        """
        Validate that the title is not empty and has minimum length.
        """
        if not value or len(value.strip()) < 3:
            raise serializers.ValidationError("Title must be at least 3 characters long.")
        return value
    
    def validate_priority(self, value):
        """
        Validate priority field.
        """
        valid_priorities = ['low', 'medium', 'high']
        if value not in valid_priorities:
            raise serializers.ValidationError(f"Priority must be one of: {', '.join(valid_priorities)}")
        return value
EOF

# Create api/views.py
cat > api/views.py << 'EOF'
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Task
from .serializers import TaskSerializer

@api_view(['GET'])
def api_overview(request):
    """
    API Overview endpoint that lists all available endpoints.
    """
    api_urls = {
        'Overview': '/api/',
        'List All Tasks': '/api/tasks/',
        'Get Single Task': '/api/tasks/<int:pk>/',
        'Create Task': '/api/tasks/create/',
        'Update Task': '/api/tasks/<int:pk>/update/',
        'Delete Task': '/api/tasks/<int:pk>/delete/',
    }
    return Response(api_urls)


@api_view(['GET'])
def get_tasks(request):
    """
    Retrieve all tasks from the database.
    """
    tasks = Task.objects.all()
    serializer = TaskSerializer(tasks, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def get_task(request, pk):
    """
    Retrieve a single task by its primary key.
    """
    try:
        task = Task.objects.get(id=pk)
    except Task.DoesNotExist:
        return Response(
            {'error': 'Task not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    serializer = TaskSerializer(task, many=False)
    return Response(serializer.data)


@api_view(['POST'])
def create_task(request):
    """
    Create a new task in the database.
    """
    serializer = TaskSerializer(data=request.data)
    
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['PUT'])
def update_task(request, pk):
    """
    Update an existing task.
    """
    try:
        task = Task.objects.get(id=pk)
    except Task.DoesNotExist:
        return Response(
            {'error': 'Task not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    serializer = TaskSerializer(instance=task, data=request.data, partial=True)
    
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['DELETE'])
def delete_task(request, pk):
    """
    Delete a task from the database.
    """
    try:
        task = Task.objects.get(id=pk)
    except Task.DoesNotExist:
        return Response(
            {'error': 'Task not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    task.delete()
    return Response(
        {'message': 'Task deleted successfully'},
        status=status.HTTP_204_NO_CONTENT
    )
EOF

# Create api/urls.py
cat > api/urls.py << 'EOF'
from django.urls import path
from . import views

urlpatterns = [
    # Overview endpoint
    path('', views.api_overview, name='api-overview'),
    
    # Task endpoints
    path('tasks/', views.get_tasks, name='get-tasks'),
    path('tasks/<int:pk>/', views.get_task, name='get-task'),
    path('tasks/create/', views.create_task, name='create-task'),
    path('tasks/<int:pk>/update/', views.update_task, name='update-task'),
    path('tasks/<int:pk>/delete/', views.delete_task, name='delete-task'),
]
EOF

# Create api/admin.py
cat > api/admin.py << 'EOF'
from django.contrib import admin
from .models import Task

@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    """
    Admin configuration for Task model.
    """
    list_display = ['id', 'title', 'priority', 'completed', 'created_at', 'updated_at']
    list_filter = ['completed', 'priority', 'created_at']
    search_fields = ['title', 'description']
    list_editable = ['completed', 'priority']
    ordering = ['-created_at']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('Task Information', {
            'fields': ('title', 'description')
        }),
        ('Status', {
            'fields': ('completed', 'priority')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
EOF

# Create api/apps.py
cat > api/apps.py << 'EOF'
from django.apps import AppConfig

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    verbose_name = 'Task Management API'
EOF

# Create api/tests.py
cat > api/tests.py << 'EOF'
from django.test import TestCase
from rest_framework.test import APITestCase
from rest_framework import status
from django.urls import reverse
from .models import Task

class TaskModelTest(TestCase):
    """Test cases for Task model"""
    
    def setUp(self):
        self.task = Task.objects.create(
            title="Test Task",
            description="Test Description",
            priority="high"
        )
    
    def test_task_creation(self):
        """Test that task is created correctly"""
        self.assertEqual(self.task.title, "Test Task")
        self.assertEqual(self.task.priority, "high")
        self.assertFalse(self.task.completed)
    
    def test_task_str(self):
        """Test string representation of task"""
        self.assertEqual(str(self.task), "Test Task")


class TaskAPITest(APITestCase):
    """Test cases for Task API endpoints"""
    
    def setUp(self):
        self.task = Task.objects.create(
            title="Test Task",
            description="Test Description",
            priority="medium"
        )
    
    def test_get_tasks(self):
        """Test retrieving all tasks"""
        url = reverse('get-tasks')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
    
    def test_get_single_task(self):
        """Test retrieving a single task"""
        url = reverse('get-task', args=[self.task.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['title'], 'Test Task')
    
    def test_create_task(self):
        """Test creating a new task"""
        url = reverse('create-task')
        data = {
            'title': 'New Task',
            'description': 'New Description',
            'priority': 'low'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Task.objects.count(), 2)
    
    def test_update_task(self):
        """Test updating an existing task"""
        url = reverse('update-task', args=[self.task.id])
        data = {
            'title': 'Updated Task',
            'completed': True
        }
        response = self.client.put(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.task.refresh_from_db()
        self.assertEqual(self.task.title, 'Updated Task')
        self.assertTrue(self.task.completed)
    
    def test_delete_task(self):
        """Test deleting a task"""
        url = reverse('delete-task', args=[self.task.id])
        response = self.client.delete(url)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(Task.objects.count(), 0)
    
    def test_get_nonexistent_task(self):
        """Test retrieving a task that doesn't exist"""
        url = reverse('get-task', args=[9999])
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
EOF

# Create SETUP_INSTRUCTIONS.md
cat > SETUP_INSTRUCTIONS.md << 'EOF'
# Setup Instructions

## Quick Start

1. Run the setup script:
```bash
bash setup_project.sh
```

2. Navigate to the project:
```bash
cd task-management-api
```

3. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

6. Create superuser (optional):
```bash
python manage.py createsuperuser
```

7. Start the server:
```bash
python manage.py runserver
```

## Git Setup

Initialize git and push to GitHub:

```bash
cd task-management-api
git init
git add .
git commit -m "Initial commit: Django REST API for Task Management"
git branch -M main
git remote add origin https://github.com/yourusername/task-management-api.git
git push -u origin main
```

## Testing

Run tests:
```bash
python manage.py test
```

## API Endpoints

- GET `/api/` - API Overview
- GET `/api/tasks/` - List all tasks
- GET `/api/tasks/<id>/` - Get single task
- POST `/api/tasks/create/` - Create task
- PUT `/api/tasks/<id>/update/` - Update task
- DELETE `/api/tasks/<id>/delete/` - Delete task

Visit `http://127.0.0.1:8000/api/` to see the API!
EOF

# Make manage.py executable
chmod +x manage.py

echo ""
echo "✅ Project structure created successfully!"
echo ""
echo "Next steps:"
echo "1. cd task-management-api"
echo "2. python -m venv venv"
echo "3. source venv/bin/activate"
echo "4. pip install -r requirements.txt"
echo "5. python manage.py makemigrations"
echo "6. python manage.py migrate"
echo "7. python manage.py runserver"
echo ""
echo "To push to GitHub:"
echo "1. git init"
echo "2. git add ."
echo "3. git commit -m 'Initial commit: Django REST API'"
echo "4. git remote add origin YOUR_GITHUB_REPO_URL"
echo "5. git push -u origin main"
