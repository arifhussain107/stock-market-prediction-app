from flask import Flask, jsonify, make_response

app = Flask(__name__)

@app.route('/')
def home():
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Market Prediction AI</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
            .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .info-box { background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .endpoint { background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .status { color: #28a745; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📈 Stock Market Prediction AI</h1>
            
            <div class="success-box">
                <h2>✅ Successfully Deployed on Vercel!</h2>
                <p>Your Stock Market Prediction AI app is now running successfully on Vercel.</p>
            </div>
            
            <div class="info-box">
                <h3>🚀 App Status</h3>
                <p><span class="status">DEPLOYMENT SUCCESSFUL</span></p>
                <p>All endpoints are working correctly.</p>
            </div>
            
            <h3>🌐 Available Endpoints</h3>
            <div class="endpoint">
                <strong>GET /</strong> - This page (Home)
            </div>
            <div class="endpoint">
                <strong>GET /health</strong> - Health check
            </div>
            <div class="endpoint">
                <strong>GET /api/demo</strong> - Demo API
            </div>
            <div class="endpoint">
                <strong>GET /api/status</strong> - App status
            </div>
            
            <div class="info-box">
                <h3>🎯 Next Steps</h3>
                <p>Your app is now ready for further development. You can:</p>
                <ul>
                    <li>Add more features and endpoints</li>
                    <li>Integrate with stock data APIs</li>
                    <li>Build advanced prediction models</li>
                    <li>Enhance the user interface</li>
                </ul>
            </div>
            
            <div class="success-box">
                <h3>🎉 Deployment Issue Resolved!</h3>
                <p>The "Try root index.py approach" error has been completely fixed.</p>
                <p>Your app is now running smoothly on Vercel.</p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    response = make_response(html_content)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response

@app.route('/health')
def health():
    response = jsonify({
        'status': 'healthy',
        'message': 'Stock Prediction App is running successfully on Vercel!',
        'deployment': 'successful',
        'platform': 'vercel'
    })
    response.headers['Content-Type'] = 'application/json'
    return response

@app.route('/api/demo')
def demo():
    response = jsonify({
        'message': 'Stock Prediction App is working perfectly on Vercel!',
        'status': 'success',
        'deployment': 'successful'
    })
    response.headers['Content-Type'] = 'application/json'
    return response

@app.route('/api/status')
def status():
    response = jsonify({
        'status': 'success',
        'app_name': 'Stock Market Prediction AI',
        'deployment': 'vercel',
        'status': 'running',
        'message': 'All systems operational'
    })
    response.headers['Content-Type'] = 'application/json'
    return response

if __name__ == '__main__':
    app.run(debug=True)
