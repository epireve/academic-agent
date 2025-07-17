# Academic Agent Configuration System

A robust YAML-based configuration system with Pydantic validation for the academic-agent project.

## Overview

This configuration system provides:

- **YAML-based configuration** files for different environments
- **Pydantic validation** for type safety and data integrity
- **Environment-specific settings** (development, production, test)
- **Extensible configuration** structure for future needs
- **Comprehensive error handling** and validation
- **Migration tools** for existing configurations

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install pyyaml pydantic pydantic-settings
   ```

2. **Load configuration in your code**:
   ```python
   from config.config_manager import ConfigurationManager
   
   # Initialize configuration manager
   config_manager = ConfigurationManager("config")
   
   # Load configuration for specific environment
   config = config_manager.load_config("development")
   
   # Access configuration values
   print(f"Quality threshold: {config.quality_threshold}")
   print(f"Debug mode: {config.debug}")
   ```

3. **Use with agents**:
   ```python
   from config.integration import ConfiguredBaseAgent
   
   # Create configured agent
   agent = ConfiguredBaseAgent("notes_agent", config_manager)
   
   # Agent automatically uses configuration settings
   print(f"Agent timeout: {agent.get_processing_timeout()}")
   print(f"Agent quality threshold: {agent.get_quality_threshold()}")
   ```

## Configuration Files

### Base Configuration (`base.yaml`)

Contains the default configuration that applies to all environments:

```yaml
environment: development
debug: false
version: "1.0.0"
quality_threshold: 0.75
improvement_threshold: 0.3
max_improvement_cycles: 3
communication_interval: 30

agents:
  ingestion_agent:
    agent_id: ingestion_agent
    enabled: true
    max_retries: 3
    timeout: 600
    quality_threshold: 0.8
    specialized_prompt: "..."
  # ... other agents

feedback_loops:
  - source: quality_manager
    target: notes_agent
    type: quality
    interval: 300
    enabled: true
  # ... other feedback loops

# ... other configuration sections
```

### Environment-Specific Configurations

- **`development.yaml`**: Development environment settings
  - Debug mode enabled
  - Lower quality thresholds
  - More frequent feedback loops
  - Detailed logging

- **`production.yaml`**: Production environment settings
  - Debug mode disabled
  - Higher quality thresholds
  - Optimized performance settings
  - Minimal logging

- **`test.yaml`**: Test environment settings
  - Short timeouts
  - Minimal feedback loops
  - In-memory database
  - Temporary file paths

## Configuration Structure

### Core Settings

- `environment`: Current environment name
- `debug`: Debug mode flag
- `version`: Configuration version
- `quality_threshold`: Global quality threshold
- `improvement_threshold`: Improvement threshold
- `max_improvement_cycles`: Maximum improvement cycles
- `communication_interval`: Agent communication interval

### Agent Configuration

Each agent has its own configuration:

```yaml
agents:
  agent_name:
    agent_id: string          # Agent identifier
    enabled: boolean          # Whether agent is enabled
    max_retries: integer      # Maximum retry attempts
    timeout: integer          # Processing timeout (seconds)
    quality_threshold: float  # Quality threshold (0.0-1.0)
    specialized_prompt: string # Agent-specific prompt
```

### Feedback Loops

Configure inter-agent communication:

```yaml
feedback_loops:
  - source: string      # Source agent ID
    target: string      # Target agent ID
    type: string        # Feedback type
    interval: integer   # Interval in seconds
    enabled: boolean    # Whether loop is enabled
```

### Processing Configuration

```yaml
processing:
  max_concurrent_agents: integer     # Maximum concurrent agents
  processing_timeout: integer        # Processing timeout (seconds)
  retry_on_failure: boolean          # Whether to retry on failure
  preserve_intermediate_results: boolean  # Preserve intermediate results
  batch_size: integer               # Batch size for processing
  enable_checkpointing: boolean     # Enable checkpointing
```

### Logging Configuration

```yaml
logging:
  level: string              # Log level (DEBUG, INFO, WARNING, ERROR)
  format: string            # Log format string
  file_enabled: boolean     # Enable file logging
  console_enabled: boolean  # Enable console logging
  log_dir: string          # Log directory path
  max_file_size: integer   # Maximum log file size (bytes)
  backup_count: integer    # Number of backup log files
```

### Path Configuration

```yaml
paths:
  input_dir: string       # Input directory
  output_dir: string      # Output directory
  processed_dir: string   # Processed files directory
  analysis_dir: string    # Analysis directory
  outlines_dir: string    # Outlines directory
  notes_dir: string       # Notes directory
  metadata_dir: string    # Metadata directory
  temp_dir: string        # Temporary files directory
```

## Environment Variables

The system automatically loads environment variables from `.env` files:

```bash
# API Keys
GROQ_API_KEY=your_groq_api_key_here
HF_API_KEY=hf_your_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Environment settings
ENVIRONMENT=development
DEBUG_MODE=true

# Processing settings
PDF_PROCESSING_DEVICE=mps
DEFAULT_OUTPUT_DIR=./output

# Database settings
DATABASE_URL=postgresql://user:pass@localhost/db
```

## Command Line Tools

### Configuration Validation

```bash
# Validate all configurations
python config/validate_config.py

# Validate specific environment
python config/validate_config.py --environment development

# Validate specific file
python config/validate_config.py --file config/base.yaml

# Validate only YAML syntax
python config/validate_config.py --syntax-only
```

### Configuration Utilities

```bash
# Create sample configuration files
python config/config_utils.py create

# Validate configuration
python config/config_utils.py validate --environment development

# Export configuration to JSON
python config/config_utils.py export --environment production --format json

# Optimize configuration for environment
python config/config_utils.py optimize --environment production

# Generate configuration documentation
python config/config_utils.py document --environment development
```

### Migration from JSON

```bash
# Migrate existing JSON configuration to YAML
python config/migrate_config.py
```

## Testing the Configuration System

Run the test script to verify everything works:

```bash
python test_config_system.py
```

This will test:
- Basic configuration loading
- Environment-specific configurations
- Agent configuration access
- Feedback loop configurations
- Path creation functionality
- Configuration validation

## Integration with Existing Code

### Using ConfiguredBaseAgent

Replace your existing BaseAgent with ConfiguredBaseAgent:

```python
from config.integration import ConfiguredBaseAgent

class MyAgent(ConfiguredBaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        
    def _process_message(self, message):
        # Your message processing logic
        return True
        
    def check_quality(self, content):
        # Your quality checking logic
        return 0.8
```

### Direct Configuration Access

```python
from config.config_manager import ConfigurationManager

# Load configuration
config_manager = ConfigurationManager()
config = config_manager.load_config("development")

# Access specific settings
quality_threshold = config.quality_threshold
agent_config = config.get_agent_config("notes_agent")
feedback_loops = config.get_feedback_loops_for_agent("notes_agent")
```

## Error Handling

The configuration system provides comprehensive error handling:

- **ConfigurationError**: Thrown for configuration-related issues
- **ValidationError**: Thrown for Pydantic validation failures
- **FileNotFoundError**: Thrown when configuration files are missing
- **YAMLError**: Thrown for YAML parsing errors

Example error handling:

```python
from config.config_manager import ConfigurationManager, ConfigurationError

try:
    config_manager = ConfigurationManager()
    config = config_manager.load_config("development")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle configuration error
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected error
```

## Extending the Configuration System

### Adding New Configuration Sections

1. **Update the Pydantic model** in `config_manager.py`:

```python
class MyCustomConfig(BaseModel):
    enabled: bool = Field(default=True)
    setting1: str = Field(default="value1")
    setting2: int = Field(default=42)

class AcademicAgentConfig(BaseModel):
    # ... existing fields
    my_custom: MyCustomConfig = Field(default_factory=MyCustomConfig)
```

2. **Add to YAML configuration files**:

```yaml
my_custom:
  enabled: true
  setting1: "custom_value"
  setting2: 100
```

3. **Access in your code**:

```python
config = config_manager.load_config("development")
my_setting = config.my_custom.setting1
```

### Adding New Environments

1. **Create a new YAML file** (e.g., `config/staging.yaml`)
2. **Add environment-specific overrides**
3. **Load the configuration**:

```python
config = config_manager.load_config("staging")
```

## Best Practices

1. **Environment Separation**: Keep environment-specific settings in separate files
2. **Validation**: Always validate configuration before use
3. **Documentation**: Document all configuration options
4. **Defaults**: Provide sensible defaults for all settings
5. **Error Handling**: Handle configuration errors gracefully
6. **Testing**: Test configuration loading in your test suite
7. **Security**: Never commit sensitive data (API keys, passwords) to version control

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   - Install required dependencies: `pip install pyyaml pydantic pydantic-settings`

2. **Configuration validation errors**:
   - Run `python config/validate_config.py` to identify issues
   - Check YAML syntax with `--syntax-only` flag

3. **Environment variables not loading**:
   - Ensure `.env` file exists in the project root
   - Check file permissions and encoding

4. **Path creation failures**:
   - Ensure proper write permissions
   - Check disk space availability

### Getting Help

- Check the validation output for specific error messages
- Run tests to verify system functionality
- Use the configuration utilities for debugging
- Review the sample configurations for examples

## Migration Notes

If you're migrating from the existing JSON configuration:

1. **Backup your existing configuration**:
   ```bash
   cp config/academic_agent_config.json config/academic_agent_config.json.backup
   ```

2. **Run the migration script**:
   ```bash
   python config/migrate_config.py
   ```

3. **Validate the new configuration**:
   ```bash
   python config/validate_config.py
   ```

4. **Test your application** with the new configuration system

5. **Update your code** to use the new configuration API

The migration script will:
- Convert JSON to YAML format
- Create environment-specific files
- Preserve all existing settings
- Add new configuration options with defaults
- Create a backup of the original file