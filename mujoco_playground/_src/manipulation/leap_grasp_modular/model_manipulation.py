from base import get_assets


assets = get_assets()

print(f"assets::type::{type(assets)}")
print(f"assets::keys::{assets.keys()}")
print(f"assets::leap_mount.obj::{assets['leap_mount.obj']}")
print(f"assets::leap_mount.obj::type::{type(assets['leap_mount.obj'])}")


