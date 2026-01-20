#!/usr/bin/env python3
"""
HOSTINGER BUSINESS PLAN DEPLOYMENT AUTOMATION
Automated deployment to Hostinger shared hosting with full stack setup
"""

import os
import sys
import subprocess
import paramiko
from pathlib import Path
from dotenv import load_dotenv
import time

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{'=' * 80}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {text}{Colors.NC}")
    print(f"{'=' * 80}\n")

def print_step(step, text):
    print(f"{Colors.BLUE}[{step}]{Colors.NC} {text}")

def print_success(text):
    print(f"{Colors.GREEN}✓{Colors.NC} {text}")

def print_error(text):
    print(f"{Colors.RED}✗ ERROR:{Colors.NC} {text}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ WARNING:{Colors.NC} {text}")

class HostingerDeployer:
    """Automated Hostinger deployment manager"""

    def __init__(self):
        load_dotenv()

        # Hostinger credentials from .env
        self.ssh_host = os.getenv('HOSTINGER_SSH_HOST', '46.202.197.97')
        self.ssh_user = os.getenv('HOSTINGER_SSH_USER', 'u472811699')
        self.ssh_password = os.getenv('HOSTINGER_SSH_PASSWORD')
        self.ssh_port = int(os.getenv('HOSTINGER_SSH_PORT', '65002'))

        # Deployment paths
        self.local_path = Path(__file__).parent
        self.remote_base = '/home/u472811699'
        self.remote_app = f'{self.remote_base}/ultimate-orchestrator'
        self.remote_public = f'{self.remote_base}/public_html'

        # SSH client
        self.ssh = None
        self.sftp = None

    def connect_ssh(self):
        """Establish SSH connection to Hostinger"""
        print_step("1/10", "Connecting to Hostinger via SSH...")

        try:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            self.ssh.connect(
                hostname=self.ssh_host,
                port=self.ssh_port,
                username=self.ssh_user,
                password=self.ssh_password,
                timeout=30
            )

            self.sftp = self.ssh.open_sftp()
            print_success(f"Connected to {self.ssh_host}:{self.ssh_port}")
            return True

        except Exception as e:
            print_error(f"SSH connection failed: {e}")
            return False

    def execute_remote(self, command, silent=False):
        """Execute command on remote server"""
        try:
            stdin, stdout, stderr = self.ssh.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()

            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')

            if not silent:
                if output:
                    print(output)
                if error and exit_status != 0:
                    print_error(error)

            return exit_status == 0, output, error

        except Exception as e:
            print_error(f"Command execution failed: {e}")
            return False, "", str(e)

    def setup_remote_environment(self):
        """Setup Python virtual environment and install dependencies"""
        print_step("2/10", "Setting up remote Python environment...")

        commands = [
            # Create app directory
            f"mkdir -p {self.remote_app}",

            # Setup Python virtual environment
            f"cd {self.remote_app} && python3 -m venv venv",

            # Upgrade pip
            f"cd {self.remote_app} && source venv/bin/activate && pip install --upgrade pip",

            # Install core dependencies
            f"cd {self.remote_app} && source venv/bin/activate && pip install fastapi uvicorn websockets",
        ]

        for cmd in commands:
            success, output, error = self.execute_remote(cmd)
            if not success:
                print_warning(f"Command had issues: {cmd}")

        print_success("Python environment configured")

    def upload_files(self):
        """Upload application files to Hostinger"""
        print_step("3/10", "Uploading application files...")

        # Files to upload
        files_to_upload = [
            'api_server.py',
            'meta_orchestrator.py',
            'voice_interface.py',
            'new_ultra_agents.py',
            'database_config.py',
            'unified_config.py',
            'livekit_voice.py',  # Will create this
            'requirements.txt',
            '.env'
        ]

        try:
            # Ensure remote directory exists
            self.execute_remote(f"mkdir -p {self.remote_app}", silent=True)

            uploaded = 0
            for filename in files_to_upload:
                local_file = self.local_path / filename

                if local_file.exists():
                    remote_file = f"{self.remote_app}/{filename}"
                    self.sftp.put(str(local_file), remote_file)
                    uploaded += 1
                else:
                    print_warning(f"File not found: {filename}")

            print_success(f"Uploaded {uploaded} files")
            return True

        except Exception as e:
            print_error(f"File upload failed: {e}")
            return False

    def install_dependencies(self):
        """Install Python dependencies from requirements.txt"""
        print_step("4/10", "Installing Python dependencies...")

        cmd = f"cd {self.remote_app} && source venv/bin/activate && pip install -r requirements.txt"
        success, output, error = self.execute_remote(cmd)

        if success:
            print_success("Dependencies installed")
        else:
            print_warning("Some dependencies may have failed - check compatibility")

        return True

    def setup_supervisor(self):
        """Setup process manager to keep app running"""
        print_step("5/10", "Configuring process manager...")

        supervisor_config = f"""[program:ultimate-orchestrator]
command={self.remote_app}/venv/bin/python {self.remote_app}/api_server.py
directory={self.remote_app}
user={self.ssh_user}
autostart=true
autorestart=true
stderr_logfile={self.remote_app}/logs/error.log
stdout_logfile={self.remote_app}/logs/access.log
environment=PATH="{self.remote_app}/venv/bin"
"""

        # Create supervisor config
        config_path = f"{self.remote_app}/supervisor.conf"

        # Upload config (create temp file first)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(supervisor_config)
            temp_path = f.name

        self.sftp.put(temp_path, config_path)
        os.unlink(temp_path)

        # Create logs directory
        self.execute_remote(f"mkdir -p {self.remote_app}/logs", silent=True)

        print_success("Process manager configured")

    def setup_nginx_proxy(self):
        """Setup Nginx reverse proxy for the API"""
        print_step("6/10", "Configuring Nginx reverse proxy...")

        nginx_config = f"""# Ultimate Orchestrator API
location /api/ {{
    proxy_pass http://127.0.0.1:8000/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}}

# WebSocket for LiveKit
location /ws/ {{
    proxy_pass http://127.0.0.1:8000/ws/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 86400;
}}
"""

        # Save to remote
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(nginx_config)
            temp_path = f.name

        nginx_path = f"{self.remote_base}/.nginx/ultimate-orchestrator.conf"
        self.execute_remote(f"mkdir -p {self.remote_base}/.nginx", silent=True)

        try:
            self.sftp.put(temp_path, nginx_path)
            os.unlink(temp_path)
            print_success("Nginx proxy configured")
        except:
            print_warning("Nginx config saved - may need manual setup")

    def upload_frontend(self):
        """Upload futuristic frontend to public_html"""
        print_step("7/10", "Uploading futuristic frontend...")

        # Will upload the frontend files we create
        frontend_files = [
            'index.html',
            'style.css',
            'app.js',
            'livekit-client.js'
        ]

        # Create public_html if needed
        self.execute_remote(f"mkdir -p {self.remote_public}", silent=True)

        uploaded = 0
        for filename in frontend_files:
            local_file = self.local_path / 'frontend' / filename

            if local_file.exists():
                remote_file = f"{self.remote_public}/{filename}"
                self.sftp.put(str(local_file), remote_file)
                uploaded += 1

        print_success(f"Frontend uploaded ({uploaded} files)")

    def start_application(self):
        """Start the FastAPI application"""
        print_step("8/10", "Starting application...")

        # Kill any existing process
        self.execute_remote("pkill -f api_server.py", silent=True)

        # Start in background using nohup
        start_cmd = f"cd {self.remote_app} && nohup venv/bin/python api_server.py > logs/app.log 2>&1 &"
        self.execute_remote(start_cmd, silent=True)

        # Give it time to start
        time.sleep(3)

        # Check if running
        success, output, _ = self.execute_remote("pgrep -f api_server.py", silent=True)

        if success and output.strip():
            print_success(f"Application started (PID: {output.strip()})")
            return True
        else:
            print_warning("Application may not have started - check logs")
            return False

    def verify_deployment(self):
        """Verify deployment is working"""
        print_step("9/10", "Verifying deployment...")

        # Check process
        success, output, _ = self.execute_remote("pgrep -f api_server.py", silent=True)
        if success and output.strip():
            print_success("Process is running")
        else:
            print_error("Process not found")
            return False

        # Check logs
        success, output, _ = self.execute_remote(f"tail -20 {self.remote_app}/logs/app.log", silent=True)
        if "Uvicorn running" in output or "Application startup complete" in output:
            print_success("Application started successfully")
        else:
            print(f"\n{Colors.YELLOW}Recent logs:{Colors.NC}")
            print(output)

        return True

    def print_summary(self):
        """Print deployment summary"""
        print_step("10/10", "Deployment complete!")

        # Get domain from env or use IP
        domain = os.getenv('HOSTINGER_DOMAIN', self.ssh_host)

        print_header("🚀 DEPLOYMENT SUMMARY")

        print(f"{Colors.BOLD}Application URLs:{Colors.NC}")
        print(f"  • API Endpoint: http://{domain}/api/")
        print(f"  • API Docs: http://{domain}/api/docs")
        print(f"  • Frontend: http://{domain}/")
        print(f"  • WebSocket: ws://{domain}/ws/")
        print()

        print(f"{Colors.BOLD}Server Info:{Colors.NC}")
        print(f"  • Host: {self.ssh_host}:{self.ssh_port}")
        print(f"  • User: {self.ssh_user}")
        print(f"  • Path: {self.remote_app}")
        print()

        print(f"{Colors.BOLD}Management Commands:{Colors.NC}")
        print(f"  • View logs: ssh -p {self.ssh_port} {self.ssh_user}@{self.ssh_host} 'tail -f {self.remote_app}/logs/app.log'")
        print(f"  • Restart: ssh -p {self.ssh_port} {self.ssh_user}@{self.ssh_host} 'pkill -f api_server.py && cd {self.remote_app} && nohup venv/bin/python api_server.py &'")
        print(f"  • Status: ssh -p {self.ssh_port} {self.ssh_user}@{self.ssh_host} 'pgrep -f api_server.py'")
        print()

        print(f"{Colors.BOLD}Next Steps:{Colors.NC}")
        print(f"  1. Test API: curl http://{domain}/api/health")
        print(f"  2. Open frontend in browser")
        print(f"  3. Test voice interface via LiveKit")
        print(f"  4. Monitor logs for any issues")
        print()

    def deploy(self):
        """Execute full deployment"""
        print_header("🚀 HOSTINGER BUSINESS PLAN DEPLOYMENT")

        try:
            # Connect
            if not self.connect_ssh():
                return False

            # Deploy
            self.setup_remote_environment()
            self.upload_files()
            self.install_dependencies()
            self.setup_supervisor()
            self.setup_nginx_proxy()
            self.upload_frontend()
            self.start_application()
            self.verify_deployment()
            self.print_summary()

            return True

        except Exception as e:
            print_error(f"Deployment failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Cleanup
            if self.sftp:
                self.sftp.close()
            if self.ssh:
                self.ssh.close()

def main():
    """Main deployment entry point"""
    deployer = HostingerDeployer()
    success = deployer.deploy()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
