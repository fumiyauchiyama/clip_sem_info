```
[rank5]: Traceback (most recent call last):
[rank5]:   File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
[rank5]:     return _run_code(code, main_globals, None,
[rank5]:   File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
[rank5]:     exec(code, run_globals)
[rank5]:   File "/groups/gag51404/fumiyau/repos/clip_sem_info/src/open_clip_train/main.py", line 555, in <module>
[rank5]:     main(sys.argv[1:])
[rank5]:   File "/groups/gag51404/fumiyau/repos/clip_sem_info/src/open_clip_train/main.py", line 223, in main
[rank5]:     model, preprocess_train, preprocess_val = create_model_and_transforms(
[rank5]:   File "/groups/gag51404/fumiyau/repos/clip_sem_info/src/open_clip/factory.py", line 502, in create_model_and_transforms
[rank5]:     model = create_model(
[rank5]:   File "/groups/gag51404/fumiyau/repos/clip_sem_info/src/open_clip/factory.py", line 335, in create_model
[rank5]:     assert False, 'pretrained image towers currently only supported for timm models'
[rank5]: AssertionError: pretrained image towers currently only supported for timm models
```