mod frb_generated; /* AUTO INJECTED BY flutter_rust_bridge. This line may not be accurate, and you can change it according to your needs. */
mod aec_manager;
mod asr_manager;
mod config;
mod denoiser_manager;
mod duplex_audio;
mod engine;
mod engine_handle;
mod errors;
mod realtime_pipeline;
mod tts_sink;
mod tts_fx;
mod tts_manager;
mod vad_manager;
pub mod wakeword;
pub mod viot;

pub mod ffi;

pub use config::EngineConfig;
pub use engine::VocomEngine;
pub use engine_handle::{EngineCommand, EngineEvent, VocomEngineHandle};

#[cfg(target_os = "android")]
static ANDROID_CONTEXT_GLOBAL_REF: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_example_vocom_1demo_MainActivity_initNativeAudioContext(
	env: *mut jni_sys::JNIEnv,
	_clazz: jni_sys::jclass,
	context: jni_sys::jobject,
) {
	if ANDROID_CONTEXT_GLOBAL_REF.get().is_some() {
		return;
	}

	if env.is_null() || context.is_null() {
		eprintln!("initNativeAudioContext called with null JNI pointers");
		return;
	}

	let new_global_ref = unsafe { (**env).NewGlobalRef };
	let Some(new_global_ref_fn) = new_global_ref else {
		eprintln!("JNI NewGlobalRef function pointer is null");
		return;
	};

	let global_context = unsafe { new_global_ref_fn(env, context) };
	if global_context.is_null() {
		eprintln!("failed to create global Android context reference");
		return;
	}

	let mut vm: *mut jni_sys::JavaVM = std::ptr::null_mut();
	let get_java_vm = unsafe { (**env).GetJavaVM };
	let Some(get_java_vm_fn) = get_java_vm else {
		eprintln!("JNI GetJavaVM function pointer is null");
		return;
	};

	let result = unsafe { get_java_vm_fn(env, &mut vm as *mut _) };
	if result != 0 || vm.is_null() {
		eprintln!("failed to resolve JavaVM in initNativeAudioContext: {result}");
		return;
	}

	unsafe {
		ndk_context::initialize_android_context(
			vm as *mut std::ffi::c_void,
			global_context as *mut std::ffi::c_void,
		);
	}

	let _ = ANDROID_CONTEXT_GLOBAL_REF.set(global_context as usize);
}
