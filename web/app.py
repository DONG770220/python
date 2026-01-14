import os
import json
import faiss
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_image import center_crop
from dinov2_numpy import Dinov2Numpy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 历史记录文件
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'history.json')

# 全局变量
model = None
index = None
paths = None


def load_model():
    """加载模型和索引"""
    global model, index, paths

    weights_path = os.path.join(PROJECT_ROOT, "vit-dinov2-base.npz")
    index_path = os.path.join(PROJECT_ROOT, "features", "index.faiss")
    paths_path = os.path.join(PROJECT_ROOT, "features", "paths.npy")

    print("正在加载模型...")
    weights = np.load(weights_path, allow_pickle=True)
    model = Dinov2Numpy(weights)

    print("正在加载索引...")
    index = faiss.read_index(index_path)

    print("正在加载路径...")
    paths = np.load(paths_path, allow_pickle=True)

    print(f"加载完成！共有 {len(paths)} 张图片")


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_history():
    """加载历史记录"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_history(history):
    """保存历史记录"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def add_to_history(query_image, results_count):
    """添加查询到历史记录"""
    history = load_history()
    record = {
        'id': len(history) + 1,
        'query_image': query_image,
        'results_count': results_count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    history.insert(0, record)
    # 只保留最近100条记录
    history = history[:100]
    save_history(history)


def search_similar(image_path, top_k=50):
    """搜索相似图片"""
    global model, index, paths

    # 预处理查询图片
    preprocessed = center_crop(image_path)

    # 提取特征
    query_feature = model(preprocessed).astype("float32")

    # 归一化
    faiss.normalize_L2(query_feature)

    # 搜索
    distances, indices = index.search(query_feature, top_k)

    # 构建结果
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        results.append({
            'rank': i + 1,
            'image': paths[idx],
            'similarity': float(dist) * 100  # 转换为百分比
        })

    return results


@app.route('/')
def home():
    """首页"""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """处理搜索请求"""
    if 'image' not in request.files:
        return redirect(url_for('home'))

    file = request.files['image']

    if file.filename == '':
        return redirect(url_for('home'))

    if file and allowed_file(file.filename):
        # 安全保存文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # 搜索相似图片
            results = search_similar(filepath)

            # 添加到历史记录
            add_to_history(filename, len(results))

            return render_template('results.html',
                                   query_image=filename,
                                   results=results)
        except Exception as e:
            return render_template('index.html', error=str(e))

    return redirect(url_for('home'))


@app.route('/history')
def history():
    """历史记录页面"""
    records = load_history()
    return render_template('history.html', records=records)


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """清除历史记录"""
    save_history([])
    return redirect(url_for('history'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传的图片"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/images/<filename>')
def image_file(filename):
    """提供数据集中的图片"""
    images_dir = os.path.join(PROJECT_ROOT, "assignments", "1")
    return send_from_directory(images_dir, filename)


@app.route('/api/search', methods=['POST'])
def api_search():
    """API搜索接口"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    filename = timestamp + filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        results = search_similar(filepath)
        add_to_history(filename, len(results))
        return jsonify({
            'query_image': filename,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model()
    print("\n" + "=" * 50)
    print("🚀 图像相似检索系统已启动！")
    print("📍 访问地址: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)