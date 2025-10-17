import os
import tempfile
import shutil
import zipfile
import json
from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd

class FileUtils:
    """Utility functions for file operations"""
    
    @staticmethod
    def create_temp_directory(prefix: str = "pbi_temp") -> str:
        """Create a temporary directory"""
        return tempfile.mkdtemp(prefix=prefix)
    
    @staticmethod
    def cleanup_temp_directory(temp_dir: str):
        """Clean up temporary directory"""
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str, encoding: str = 'utf-8'):
        """Save data as JSON file"""
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_json(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """Load JSON file"""
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    
    @staticmethod
    def create_zip_file(source_dir: str, output_path: str):
        """Create a ZIP file from directory"""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arc_name)
    
    @staticmethod
    def extract_zip_file(zip_path: str, extract_to: str):
        """Extract ZIP file to directory"""
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_to)
    
    @staticmethod
    def read_zip_contents(zip_path: str) -> Dict[str, bytes]:
        """Read contents of ZIP file into memory"""
        contents = {}
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            for file_name in zipf.namelist():
                contents[file_name] = zipf.read(file_name)
        return contents
    
    @staticmethod
    def write_zip_contents(contents: Dict[str, bytes], output_path: str):
        """Write contents to ZIP file"""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_name, file_content in contents.items():
                zipf.writestr(file_name, file_content)
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Get file extension"""
        return Path(file_path).suffix.lower()
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes"""
        return os.path.getsize(file_path)
    
    @staticmethod
    def ensure_directory_exists(directory: str):
        """Ensure directory exists, create if it doesn't"""
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def dataset_to_csv_string(dataset: pd.DataFrame) -> str:
        """Convert pandas DataFrame to CSV string"""
        return dataset.to_csv(index=False)
    
    @staticmethod
    def dataset_to_csv_bytes(dataset: pd.DataFrame) -> bytes:
        """Convert pandas DataFrame to CSV bytes"""
        return dataset.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def validate_file_type(file_path: str, allowed_extensions: list) -> bool:
        """Validate if file type is allowed"""
        extension = FileUtils.get_file_extension(file_path)
        return extension in [ext.lower() for ext in allowed_extensions]
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file operations"""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        
        return filename.strip()
    
    @staticmethod
    def get_safe_path(base_path: str, filename: str) -> str:
        """Get safe file path by sanitizing filename"""
        safe_filename = FileUtils.sanitize_filename(filename)
        return os.path.join(base_path, safe_filename)
    
    @staticmethod
    def copy_file(source: str, destination: str):
        """Copy file from source to destination"""
        shutil.copy2(source, destination)
    
    @staticmethod
    def move_file(source: str, destination: str):
        """Move file from source to destination"""
        shutil.move(source, destination)
    
    @staticmethod
    def delete_file(file_path: str):
        """Delete file if it exists"""
        if os.path.exists(file_path):
            os.remove(file_path)
    
    @staticmethod
    def read_text_file(file_path: str, encoding: str = 'utf-8') -> str:
        """Read text file content"""
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    @staticmethod
    def write_text_file(content: str, file_path: str, encoding: str = 'utf-8'):
        """Write content to text file"""
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    
    @staticmethod
    def read_binary_file(file_path: str) -> bytes:
        """Read binary file content"""
        with open(file_path, 'rb') as f:
            return f.read()
    
    @staticmethod
    def write_binary_file(content: bytes, file_path: str):
        """Write content to binary file"""
        with open(file_path, 'wb') as f:
            f.write(content)
    
    @staticmethod
    def get_directory_size(directory: str) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size
    
    @staticmethod
    def list_files_in_directory(directory: str, extension: str = None) -> list:
        """List files in directory, optionally filtered by extension"""
        files = []
        for file_path in Path(directory).iterdir():
            if file_path.is_file():
                if extension is None or file_path.suffix.lower() == extension.lower():
                    files.append(str(file_path))
        return files
    
    @staticmethod
    def create_backup_file(file_path: str) -> str:
        """Create backup of file with timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.backup_{timestamp}"
        shutil.copy2(file_path, backup_path)
        return backup_path
