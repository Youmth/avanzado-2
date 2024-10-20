'''
    Run this file to install all dependencies from the main program
'''

import subprocess
import sys
import re
from settings import REMOVE_VERSIONS  # Importa la variable de SETTINGS.py

def get_latest_version(package_name):
    """
    Get the latest version of a package available on PyPI.
    """
    result = subprocess.run(
        ['pip', 'index', 'versions', package_name],
        stdout=subprocess.PIPE,
        text=True
    )
    versions = re.findall(r'Available versions: ([^\n]+)', result.stdout)
    if versions:
        latest_version = versions[0].split(",")[0].strip()
        return latest_version
    return None

def remove_versions(lines):
    """
    Removes version numbers from each line of the requirements list.
    """
    updated_lines = []
    for line in lines:
        # Use regex to remove version information (e.g., '==x.x.x')
        updated_line = re.sub(r'==[\d\.\+]+', '', line).strip() + '\n'
        updated_lines.append(updated_line)
    return updated_lines

def update_requirements_file(file_path):
    """
    Reads a requirements.txt file, replaces 'skimage==0.0' with the correct 'scikit-image' version,
    and optionally removes version numbers if configured.
    """
    try:
        # Read the contents of the requirements.txt file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Find the latest version of scikit-image
        scikit_image_version = get_latest_version('scikit-image')

        if not scikit_image_version:
            print("Could not fetch the latest version of scikit-image.")
            return

        # Modify the requirements lines
        updated_lines = []
        for line in lines:
            # Replace the skimage entry with scikit-image
            if 'skimage' in line:
                updated_line = f"scikit-image=={scikit_image_version}\n"
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)

        # Remove version numbers if configured
        if REMOVE_VERSIONS:
            updated_lines = remove_versions(updated_lines)

        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)

        print("Updated requirements.txt successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Path to your requirements.txt file
file_path = "requirements.txt"
update_requirements_file(file_path)

# Install the packages
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
