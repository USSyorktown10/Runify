import yaml  # Or use json if your spec is JSON
from connexion.resolver import Resolver
from importlib import import_module
import os

# Load your OpenAPI spec
with open('spec/openapi.yaml', 'r') as f:  # Replace with your spec file path
    spec = yaml.safe_load(f)

# Initialize the resolver (match your app's configuration)
resolver = Resolver()  # Add pythonic_params=True or custom function_resolver if needed

patched = []
http_methods = {'get', 'post', 'put', 'delete', 'patch', 'options', 'head'}

for path, path_data in spec.get('paths', {}).items():
    for method, operation in path_data.items():
        if method.lower() in http_methods:
            operation_id = operation.get('operationId')
            if operation_id:
                try:
                    # Attempt to resolve the function
                    resolver.resolve_function_from_operation_id(operation_id)
                except Exception as e:
                    # Collect details
                    item = {'operation_id': operation_id, 'error': str(e), 'path': path, 'method': method.upper()}
                    patched.append(item)

                    # Manually split operation_id into module_name and function_name
                    parts = operation_id.rsplit('.', 1)
                    if len(parts) != 2:
                        print(f"Warning: Invalid operation_id format '{operation_id}' - skipping auto-define.")
                        continue
                    module_name, function_name = parts

                    # Auto-define the missing function with 'pass' by writing to file
                    try:
                        module = import_module(module_name)
                        if not hasattr(module, function_name):
                            # Append to existing file
                            file_path = module.__file__
                            with open(file_path, 'a', encoding='utf-8') as f:
                                f.write(f'\n\ndef {function_name}(*args, **kwargs):\n    pass\n')
                            item['file_path'] = file_path
                            item['action'] = 'appended'
                    except ImportError:
                        # Create new file assuming relative to current directory
                        file_path = module_name.replace('.', os.path.sep) + '.py'
                        dir_name = os.path.dirname(file_path)
                        if dir_name:
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(f'def {function_name}(*args, **kwargs):\n    pass\n')
                        item['file_path'] = file_path
                        item['action'] = 'created'

if patched:
    print("The following operations were missing and have been auto-defined with 'pass' in files:")
    for item in patched:
        print(
            f"- Operation ID: {item['operation_id']} (Path: {item['path']}, Method: {item['method']}) - Original Error: {item['error']} - File: {item.get('file_path', 'N/A')} ({item.get('action', 'N/A')})")
else:
    print("All operations are implemented.")