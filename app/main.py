from contextlib import asynccontextmanager
from pathlib import Path
import pathlib
import platform
import pickle
import torch
import sys
import types

# --- Plum compatibility shim ---
# Older fastai models (saved with plum<2.0 in Colab) reference submodules
# like plum._function / plum._resolver that no longer exist. We create
# lightweight stub classes so the pickle can reconstruct without errors.
try:
    import plum as _plum

    class _DictShim:
        def __init__(self, *args, **kwargs):
            self.device = 'cpu'
            if kwargs:
                self.__dict__.update(kwargs)
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
        def __getstate__(self):
            return self.__dict__
        def __call__(self, *args, **kwargs):
            return None

    class _ListShim(list):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
        def __getstate__(self):
            return self.__dict__
        def __call__(self, *args, **kwargs):
            return None

    def _make_plum_shim(sub):
        mod = types.ModuleType(f'plum.{sub}')
        mod.Resolver = type('Resolver', (_DictShim,), {})
        mod.Function = type('Function', (_DictShim,), {})
        mod.Method = type('Method', (_DictShim,), {})
        mod.MethodList = type('MethodList', (_ListShim,), {})
        mod.Signature = type('Signature', (_DictShim,), {})
        mod.OverloadedFunction = type('OverloadedFunction', (_DictShim,), {})
        mod.Dispatcher = type('Dispatcher', (_DictShim,), {})
        mod.__getattr__ = lambda name: type(name, (_DictShim,), {})
        return mod

    for _sub in ['_function', '_resolver', '_signature', '_type',
                 '_overload', '_dispatch', '_method', '_promotion', '_util']:
        sys.modules[f'plum.{_sub}'] = _make_plum_shim(_sub)
except ImportError:
    pass
# --- End plum shim ---

# Fix for loading fastai models across platforms
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    # On Linux/Docker, map WindowsPath to PosixPath
    pathlib.WindowsPath = pathlib.PosixPath

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import load_learner

from app.config import settings
from app.database import engine, Base
from app import predict as predict_module
# Remove separate router import to ensure we use the same module reference
# from app.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables and load model
    Base.metadata.create_all(bind=engine)

    model_path = Path(settings.models_dir) / "caltech_model.pkl"
    print(f"Loading model from {model_path}...")
    
    try:
        # load_learner is the robust way for fastai models
        predict_module.learn = load_learner(model_path, cpu=True)
    except Exception as e:
        print(f"load_learner failed: {e}")
        predict_module.learn = None
    
    if predict_module.learn is not None:
        print(f"Model loaded. Classes: {predict_module.learn.dls.vocab}")
    
    yield


app = FastAPI(title="Caltech-101 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_module.router)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predict_module.learn is not None}
