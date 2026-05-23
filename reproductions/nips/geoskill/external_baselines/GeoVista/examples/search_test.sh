#!/bin/bash

# export SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())')"
# export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
# Network connectivity test function
test_google_api() {
    echo "🔍 Testing Google API connectivity..."
    
    # Run curl with 3 second timeout and capture output
    local result
    result=$(curl -v https://www.googleapis.com 2>&1)
    local exit_code=$?
    
    # Check if command completed successfully and has the expected end message
    if [ $exit_code -eq 0 ] && echo "$result" | grep -q "Connection #0 to host.*left intact"; then
        echo -e "\033[1;34m✅ Network Test PASSED - Google API is reachable and responding correctly\033[0m"
        return 0
    else
        echo -e "\033[1;31m❌ Network Test FAILED - Unable to connect to Google API within 3 seconds\033[0m"
        echo -e "\033[1;31m💡 Please check your proxy settings or network connection\033[0m"
        return 1
    fi
}
# Run the network test
test_google_api
# If test fails, ask user if they want to continue
if [ $? -ne 0 ]; then
    echo -e "\033[1;33m⚠️  Do you want to continue anyway? (y/N)\033[0m"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi


# eval "$(conda shell.bash hook)"
# conda activate gpt-researcher
# conda activate
cd $(dirname $0)
set -a; source ../.env

python3 search_test.py
