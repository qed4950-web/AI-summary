#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command};
use tauri::{Manager, RunEvent};

fn spawn_api() -> Option<Child> {
    Command::new("python")
        .arg("scripts/search_api_server.py")
        .spawn()
        .ok()
}

fn main() {
    let api_child = spawn_api();

    tauri::Builder::default()
        .setup(|_app| {
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while running tauri application")
        .run(move |_app_handle, event| {
            if let RunEvent::Exit = event {
                if let Some(mut child) = api_child.as_ref().map(|c| c.try_clone().ok()).flatten() {
                    let _ = child.kill();
                }
            }
        });
}
