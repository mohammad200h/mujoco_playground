from brax import envs
from mujoco_playground import registry
import jax
import jax.numpy as jp


import cv2

if __name__ == "__main__":
  env_name = 'LeapGrasp'
  env = registry.load(env_name)

  # define the jit reset/step functions
  # jit_reset = jax.jit(env.reset)
  # jit_step = jax.jit(env.step)

  # generate_new_pose_and_velocity = jax.jit(env.generate_new_pose_and_velocity)

  # get_random_action = jax.jit(env.get_random_action)

  # cube_linear_vel_is_zero = jax.jit(env._cube_linear_vel_is_zero)
  # cube_angular_vel_is_zero = jax.jit(env._cube_angular_vel_is_zero)
  # there_is_a_contact_with_thumb = jax.jit(env._there_is_a_contact_with_thumb)
  # there_is_a_contact_with_any_of_ff_mf_rf = jax.jit(env._there_is_a_contact_with_any_of_ff_mf_rf)

  # rng = jax.random.PRNGKey(42)

  # state = jit_reset(rng)

  # random_action = get_random_action(rng)

  # new_qpos, new_qvel = generate_new_pose_and_velocity(rng)
  # print(f"new_qpos::shape::{new_qpos.shape} :: new_qpos:: {new_qpos}")
  # print("\n\n")
  # print(f"new_qvel::shape::{new_qvel.shape} :: new_qvel:: {new_qvel}")


  # print(f"cube_linear_vel_is_zero::{cube_linear_vel_is_zero(state.data)}")
  # print(f"cube_angular_vel_is_zero::{cube_angular_vel_is_zero(state.data)}")
#
  # print(f"there_is_a_contact_with_thumb::{there_is_a_contact_with_thumb(state.data)}")
  # print(f"there_is_a_contact_with_any_of_ff_mf_rf::{there_is_a_contact_with_any_of_ff_mf_rf(state.data)}")


  ########## batch #########
  # rng = jax.random.PRNGKey(49)
  # num_env = 2
  # rest_rng,action_rng = jax.random.split(rng, (2,num_env))

  # batch_state = jax.vmap(env.reset)(rest_rng)
  # batch_action = jax.vmap(lambda rng:env.get_random_action(rng)[0])(action_rng)
  # batched_state = jax.vmap(env.step)(batch_state,batch_action)



  # for q in batched_state.data.qpos:
  #   print("-----------------------------------")
  #   print(f"{q}")

  # for f_external in batched_state.data.qfrc_applied:
  #   print("------------f_external----------------")
  #   print(f"shape::{f_external.shape}")
  #   print(f"{f_external}")


  ############ reward function testing #######
  # rng = jax.random.PRNGKey(49)
  # num_env = 2
  # rest_rng,action_rng = jax.random.split(rng, (2,num_env))

  # batch_state = jax.vmap(env.reset)(rest_rng)
  # batch_action = jax.vmap(lambda rng:env.get_random_action(rng)[0])(action_rng)
  # batched_state = jax.vmap(env.step)(batch_state,batch_action)


  # hand_open_reward = jax.vmap(env.keep_hand_open_reward)(batch_state.data)

  # firm_grasp_reward = jax.vmap(env.firm_grasp_reward)(batch_state.data)

  # print(firm_grasp_reward)



  ############# mocap #################
  rng = jax.random.PRNGKey(49)
  num_env = 2
  rest_rng,action_rng = jax.random.split(rng, (2,num_env))

  batch_state = jax.vmap(env.reset)(rest_rng)
  batch_action = jax.vmap(lambda rng:env.get_random_action(rng)[0])(action_rng)
  batched_state= jax.vmap(env.step)(batch_state,batch_action)

  batched_flag = jax.vmap(env.cube_is_close_enough)(batched_state.data)


  # print("-------------mocap_close_enough--------------")
  # for close_enough in mocap_close_enough:
  #   print("-----------------------------------")
  #   print(f"{close_enough}")
  #   print(f"{close_enough.shape}")
  # print("--------End::mocap_close_enough--------------")

  # print("-------------mocap_far--------------")
  # for far in mocap_far:
  #   print("-----------------------------------")
  #   print(f"{far}")
  #   print(f"{far.shape}")
  # print("--------End::mocap_far--------------")


  # print("-------------mocap_quat--------------")
  # for quat in batched_state.data.mocap_quat:
  #   print("-----------------------------------")
  #   print(f"{quat}")
  #   print(f"{quat.shape}")
  # print("--------End::mocap_quat--------------")

  # print("-------------flag--------------")
  # for flag in batched_flag:
  #   print("-----------------------------------")
  #   print(f"{flag}")
  #   print(f"{flag.shape}")
  # print("--------End::flag--------------")


  ############# rendering #############
  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)
  get_random_action = jax.jit(lambda rng:env.get_random_action(rng)[0])
  get_open_action = jax.jit(lambda: jp.zeros(16))

  rng = jax.random.PRNGKey(42)
  rest_rng,a_rng = jax.random.split(rng)

  rollout = []


  n_episodes = 10

  for _ in range(n_episodes):
      state = jit_reset(rest_rng)
      episode_rollout = [state]  # Store episode states separately
      for _ in range(200):
          act_rng, a_rng = jax.random.split(a_rng)
          ctrl = get_open_action()
          state = jit_step(state, ctrl)
          episode_rollout.append(state)
      rollout.extend(episode_rollout)  # Add to the main rollout list


  render_every = 1
  frames = env.render(trajectory = rollout[::render_every],camera = "side")
  rewards = [s.reward for s in rollout]


  def display_video(frames, fps=30):
    for frame in frames:
        cv2.imshow("Simulation", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):  # Press 'q' to exit
            break
    cv2.destroyAllWindows()

  video_filename = "leap.mp4"
  fps = 1.0 / env.dt / render_every  # Frame rate based on the environment's timestep
  frame_size = (frames[0].shape[1], frames[0].shape[0])  # Width, Height


  fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
  video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

  display_video(frames, fps=1.0 / env.dt / render_every)


  for frame in frames:
      frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR (OpenCV format)
      video_writer.write(frame_bgr)  # Write frame to video


  video_writer.release()
  print(f"Video saved as {video_filename}")







