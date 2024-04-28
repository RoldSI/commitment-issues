use clap::{Command};
use nalgebra::Vector2;
use opencv::{
    core::{self, Mat, Point, Scalar},
    highgui,
    imgproc::{self, arrowed_line, circle, put_text},
    prelude::MatTraitConst,
    videoio::{self, VideoCapture, VideoCaptureTrait},
    types::VectorOfVectorOfPoint,
};
use rdev::{Event, EventType, Key, listen};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Instant};

const EYE_BALL_RADIUS_PIXEL_EQUIVALENT_HOR: f32 = 246.77;
const EYE_BALL_RADIUS_PIXEL_EQUIVALENT_VER: f32 = 213.1;

struct SharedState {
    setup: bool,
    ground_truth_position: Vector2<i32>,
    current_position: Vector2<i32>,
    quit_program: bool,
}

fn handle_key_press(shared_state: Arc<Mutex<SharedState>>) {
    let callback = move |event: Event| {
        let mut state = shared_state.lock().unwrap();
        if let Some(name) = event.name {
            if name == "e" {
                if !state.setup {
                    state.setup = true;
                    state.ground_truth_position = state.current_position;
                    println!("Calibration successful!");
                }
            } else if event.event_type == EventType::KeyPress(Key::Escape) {
                state.quit_program = true;
            }
        }
    };

    thread::spawn(move || {
        if let Err(err) = listen(callback) {
            println!("Error: {:?}", err);
        }
    });
}

fn classic_segmentation(image: &Mat) -> Result<(Vector2<i32>, Mat), Box<dyn std::error::Error>> {
    let mut gray_image = Mat::new_rows_cols_with_default(
        image.rows() as i32,
        image.cols() as i32,
        core::CV_8U,
        core::Scalar::all(0.0),
    )?;

    imgproc::cvt_color(image, &mut gray_image, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut threshold_image = Mat::default();
    imgproc::threshold(&gray_image, &mut threshold_image, 127.0, 255.0, imgproc::THRESH_BINARY)?;

    let mut contours = VectorOfVectorOfPoint::new();
    imgproc::find_contours(
        &mut threshold_image,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::default(),
    )?;

    let mut largest_area = 0.0;
    let mut largest_index = 0;
    for i in 0..contours.len() {
        let contour = &contours.get(i)?;
        let area = imgproc::contour_area(contour, false)?;
        if area > largest_area {
            largest_area = area;
            largest_index = i;
        }
    }

    let largest_contour = contours.get(largest_index)?;
    let moments = imgproc::moments(&largest_contour, false)?;  // Fixed: Borrowing the contour

    let centroid_x = (moments.m10 / moments.m00) as i32;
    let centroid_y = (moments.m01 / moments.m00) as i32;

    let centroid = Vector2::new(centroid_x, centroid_y);

    Ok((centroid, threshold_image))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("COMMITMENT ISSUES")
        .arg(
            clap::Arg::new("verbose")
                .short('v')
                .help("Enable status/debug output"),
        )
        .get_matches();

    let verbose = matches.contains_id("verbose");

    let shared_state = Arc::new(Mutex::new(SharedState {
        setup: false,
        ground_truth_position: Vector2::new(0, 0),
        current_position: Vector2::new(0, 0),
        quit_program: false,
    }));

    handle_key_press(Arc::clone(&shared_state));

    highgui::named_window("raw-image", highgui::WINDOW_NORMAL)?;
    highgui::named_window("masked-image", highgui::WINDOW_NORMAL)?;

    let mut frame_counter = 0;
    let start_time = Instant::now();

    let mut capture = VideoCapture::from_file("filename.avi", videoio::CAP_ANY)?;

    while !shared_state.lock().unwrap().quit_program {
        let mut frame = Mat::default();
        if !capture.read(&mut frame)? {
            break;
        }

        highgui::imshow("raw-image", &frame)?;

        let (centroid, mut threshold_image) = classic_segmentation(&frame)?;

        {
            let mut state = shared_state.lock().unwrap();
            state.current_position = centroid;

            if state.setup {
                let diff_position = state.current_position - state.ground_truth_position;

                let hor_diff_angle = (diff_position[0] as f32) / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_HOR;
                let hor_degree = hor_diff_angle.atan().to_degrees();

                let ver_diff_angle = (diff_position[1] as f32) / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_VER;
                let ver_degree = ver_diff_angle.atan().to_degrees();

                circle(
                    &mut threshold_image,
                    Point::new(state.current_position[0], state.current_position[1]),
                    10,
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    -1,
                    8,
                    0,
                );

                let goal_position = state.current_position + diff_position;

                arrowed_line(
                    &mut threshold_image,
                    Point::new(state.current_position[0], state.current_position[1]),
                    Point::new(goal_position[0], goal_position[1]),
                    Scalar::new(0.0, 0.0, 255.0, 0.0),
                    2,
                    8,
                    0,
                    0.2,
                );

                put_text(
                    &mut threshold_image,
                    &format!(
                        "hor gaze angle is {:.2}° and ver angle is {:.2}°",
                        hor_degree, ver_degree
                    ),
                    Point::new(50, 50),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    8,
                    imgproc::LINE_AA,
                    true,
                );

                highgui::imshow("masked-image", &threshold_image)?;
            } else {
                println!("Calibration not set");
            }
        }

        highgui::wait_key(100)?;
    }

    highgui::destroy_all_windows()?;

    Ok(())
}
