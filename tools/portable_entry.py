"""Process-local LightGBM UTF-8 model loading for Windows non-ASCII paths."""
from __future__ import annotations
import functools,inspect,os,runpy,sys
from pathlib import Path


def install_unicode_loader():
    import lightgbm as lgb
    original=lgb.Booster.__init__;signature=inspect.signature(original)
    @functools.wraps(original)
    def load(self,*args,**kwargs):
        bound=signature.bind(self,*args,**kwargs);filename=bound.arguments.get('model_file')
        if filename is not None and not str(filename).isascii():
            bound.arguments['model_file']=None
            bound.arguments['model_str']=Path(filename).read_text(encoding='utf-8')
        return original(*bound.args,**bound.kwargs)
    lgb.Booster.__init__=load


if __name__=='__main__':
    if len(sys.argv)<2:raise SystemExit('Usage: python portable_entry.py target_script.py [arguments]')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
    sys.dont_write_bytecode=True;install_unicode_loader()
    target=Path(sys.argv[1]).resolve();sys.argv=sys.argv[1:];sys.path.insert(0,str(target.parent))
    runpy.run_path(str(target),run_name='__main__')
