#!/usr/bin/env python3
"""
SAM3 Docker 環境測試腳本
測試 CUDA、PyTorch、ROS 和 SAM3 是否正確安裝
"""

import sys
import subprocess


def print_section(title):
    """打印章節標題"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_python():
    """檢查 Python 版本"""
    print_section("Python 版本檢查")
    print(f"Python 版本: {sys.version}")
    print(f"Python 可執行文件: {sys.executable}")
    
    required_version = (3, 12)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        print(f"✅ Python {current_version[0]}.{current_version[1]} >= {required_version[0]}.{required_version[1]}")
        return True
    else:
        print(f"❌ Python {current_version[0]}.{current_version[1]} < {required_version[0]}.{required_version[1]}")
        return False


def check_pytorch():
    """檢查 PyTorch 和 CUDA"""
    print_section("PyTorch 和 CUDA 檢查")
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"GPU 數量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  - 總記憶體: {props.total_memory / 1024**3:.2f} GB")
                print(f"  - 計算能力: {props.major}.{props.minor}")
            
            # 簡單的 CUDA 測試
            print("\n測試 CUDA 運算...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"✅ CUDA 矩陣運算成功: {z.shape}")
            
            return True
        else:
            print("❌ CUDA 不可用")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch 檢查失敗: {e}")
        return False


def check_ros():
    """檢查 ROS 環境"""
    print_section("ROS 環境檢查")
    
    import os
    
    ros_distro = os.environ.get('ROS_DISTRO')
    ros_version = os.environ.get('ROS_VERSION')
    
    print(f"ROS_DISTRO: {ros_distro}")
    print(f"ROS_VERSION: {ros_version}")
    
    if ros_distro and ros_version:
        # 嘗試運行 ROS 命令
        try:
            result = subprocess.run(
                ['ros2', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            print(f"ROS2 版本: {result.stdout.strip()}")
            print(f"✅ ROS Humble 環境正常")
            return True
        except Exception as e:
            print(f"❌ ROS 命令執行失敗: {e}")
            return False
    else:
        print("❌ ROS 環境變數未設置")
        return False


def check_dependencies():
    """檢查關鍵依賴"""
    print_section("依賴套件檢查")
    
    packages = [
        'numpy',
        'torch',
        'torchvision',
        'timm',
        'transformers',
        'huggingface_hub',
        'opencv-python',
        'matplotlib',
        'pillow',
        'tqdm',
    ]
    
    all_ok = True
    for package in packages:
        try:
            # 處理特殊的套件名稱
            import_name = package
            if package == 'opencv-python':
                import_name = 'cv2'
            elif package == 'pillow':
                import_name = 'PIL'
            
            module = __import__(import_name.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: 未安裝")
            all_ok = False
    
    return all_ok


def check_sam3():
    """檢查 SAM3 安裝"""
    print_section("SAM3 安裝檢查")
    
    try:
        # 嘗試導入 SAM3
        from sam3 import __version__
        print(f"SAM3 版本: {__version__}")
        
        # 測試主要組件
        print("\n測試 SAM3 組件導入...")
        
        components = [
            ('sam3.model_builder', 'build_sam3_image_model'),
            ('sam3.model.sam3_image', 'Sam3Image'),
            ('sam3.model.sam3_image_processor', 'Sam3Processor'),
        ]
        
        all_ok = True
        for module_name, component_name in components:
            try:
                module = __import__(module_name, fromlist=[component_name])
                getattr(module, component_name)
                print(f"✅ {module_name}.{component_name}")
            except Exception as e:
                print(f"❌ {module_name}.{component_name}: {e}")
                all_ok = False
        
        if all_ok:
            print("\n✅ SAM3 所有組件導入成功")
            return True
        else:
            print("\n❌ 部分 SAM3 組件導入失敗")
            return False
            
    except ImportError as e:
        print(f"❌ SAM3 未安裝或導入失敗: {e}")
        print("\n建議執行: pip install -e /workspace/sam3")
        return False


def check_gpu_memory():
    """檢查 GPU 記憶體"""
    print_section("GPU 記憶體檢查")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                
                # 記憶體資訊
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                free = total - reserved
                
                print(f"  總記憶體: {total:.2f} GB")
                print(f"  已保留: {reserved:.2f} GB")
                print(f"  已分配: {allocated:.2f} GB")
                print(f"  可用: {free:.2f} GB")
                
                if total < 8:
                    print(f"  ⚠️  記憶體可能不足，建議 16GB+")
            
            return True
        else:
            print("❌ 無可用 GPU")
            return False
            
    except Exception as e:
        print(f"❌ GPU 記憶體檢查失敗: {e}")
        return False


def print_summary(results):
    """打印測試摘要"""
    print_section("測試摘要")
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name:.<40} {status}")
    
    print(f"\n總計: {passed}/{total} 項測試通過")
    
    if passed == total:
        print("\n🎉 所有測試通過！SAM3 環境已準備就緒。")
        return True
    else:
        print("\n⚠️  部分測試失敗，請檢查上述錯誤信息。")
        return False


def main():
    """主函數"""
    print("=" * 60)
    print("  SAM3 Docker 環境測試")
    print("=" * 60)
    
    # 運行所有測試
    results = {
        'Python 版本': check_python(),
        'PyTorch 和 CUDA': check_pytorch(),
        'ROS 環境': check_ros(),
        '依賴套件': check_dependencies(),
        'GPU 記憶體': check_gpu_memory(),
        'SAM3 安裝': check_sam3(),
    }
    
    # 打印摘要
    success = print_summary(results)
    
    # 返回退出碼
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
