#include <memory>
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<char const *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

static std::vector<char> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
  std::vector<char> buffer(file.tellg());
  file.seekg(0, std::ios::beg);
  file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  file.close();
  return buffer;
}

class HelloTriangleApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow *window = nullptr;
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  vk::raii::Device device = nullptr;
  vk::raii::SwapchainKHR swapChain = nullptr;

  std::vector<vk::Image> swapChainImages;
  std::vector<vk::raii::ImageView> swapChainImageViews;
  vk::SurfaceFormatKHR swapChainSurfaceFormat;

  vk::Extent2D swapChainExtent;

  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;

  vk::raii::CommandPool commandPool = nullptr;
  vk::raii::CommandBuffer commandBuffer = nullptr;

  vk::raii::Semaphore presentCompleteSemaphore = nullptr;
  vk::raii::Semaphore renderFinishedSemaphore = nullptr;
  vk::raii::Fence drawFence = nullptr;

  vk::raii::Queue queue = nullptr;
  uint32_t queueIndex = ~0;

  std::vector<const char *> requiredDeviceExtension = {
      vk::KHRSwapchainExtensionName};

  void initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  }

  void initVulkan() {
    createInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createGraphicsPipeline();
    createCommandPool();
    createCommandBuffer();
    createSyncObjects();
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
    }
    device.waitIdle();
  }

  void createInstance() {
    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = vk::ApiVersion14,
    };

    // Get the required layers
    std::vector<char const *> requiredLayers;
    if (enableValidationLayers) {
      requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    // Check if the required layers are supported by the Vulkan implementation.
    auto layerProperties = context.enumerateInstanceLayerProperties();
    auto unsupportedLayerIt = std::ranges::find_if(
        requiredLayers, [&layerProperties](auto const &requiredLayer) {
          return std::ranges::none_of(
              layerProperties, [requiredLayer](auto const &layerProperty) {
                return std::strcmp(layerProperty.layerName, requiredLayer) == 0;
              });
        });
    if (unsupportedLayerIt != requiredLayers.end()) {
      throw std::runtime_error("Required layer not supported: " +
                               std::string(*unsupportedLayerIt));
    }

    // Get the required extensions.

    auto requiredExtensions = getRequiredInstanceExtensions();

    auto extensionProperties = context.enumerateInstanceExtensionProperties();

    std::cout << "Available extensions:" << std::endl;
    for (const auto &extension : extensionProperties) {
      std::cout << "\t" << extension.extensionName << std::endl;
    }

    // Check if the required extensions are supported by the Vulkan
    // implementation.
    auto unsupportedPropertyIt = std::ranges::find_if(
        requiredExtensions,
        [&extensionProperties](auto const &requiredExtension) {
          return std::ranges::none_of(
              extensionProperties,
              [requiredExtension](auto const &extensionProperty) {
                return std::strcmp(extensionProperty.extensionName,
                                   requiredExtension) == 0;
              });
        });
    if (unsupportedPropertyIt != requiredExtensions.end()) {
      throw std::runtime_error("Required extension not supported: " +
                               std::string(*unsupportedPropertyIt));
    }

    vk::InstanceCreateInfo createInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
        .ppEnabledLayerNames = requiredLayers.data(),
        .enabledExtensionCount =
            static_cast<uint32_t>(requiredExtensions.size()),
        .ppEnabledExtensionNames = requiredExtensions.data(),
    };

    instance = vk::raii::Instance(context, createInfo);
  }

  std::vector<const char *> getRequiredInstanceExtensions() {
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    return extensions;
  }

  void createSurface() {
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create window surface");
    }
    surface = vk::raii::SurfaceKHR(instance, _surface);
  }

  void pickPhysicalDevice() {
    auto physicalDevices = instance.enumeratePhysicalDevices();
    if (physicalDevices.empty()) {
      throw std::runtime_error("Failed to find a suitable physical device");
    }

    // use an ordered map to automatically sort candidates by increasing score
    std::multimap<int, vk::raii::PhysicalDevice> candidates;
    for (const auto &pd : physicalDevices) {
      auto deviceProperties = pd.getProperties();
      auto deviceFeatures = pd.getFeatures();

      uint32_t score = 0;
      // Discrete GPU
      if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
        score = 1000;
      }

      // max possible size of textures affects graphics quality
      score += deviceProperties.limits.maxImageDimension2D;

      if (!deviceFeatures.geometryShader) {
        continue;
      }

      if (!isDeviceSuitable(pd)) {
        continue;
      }

      candidates.insert(std::make_pair(score, pd));
    }

    if (!candidates.empty() && candidates.rbegin()->first > 0) {
      physicalDevice = candidates.rbegin()->second;
    } else {
      throw std::runtime_error("Failed to find a suitable physical device");
    }
  }

  bool isDeviceSuitable(vk::raii::PhysicalDevice const &physicalDevice) {
    auto deviceProperties = physicalDevice.getProperties();
    auto deviceFeatures = physicalDevice.getFeatures();

    // Check if the physicalDevice supports the Vulkan 1.3 API version
    bool supportsVulkan1_3 = deviceProperties.apiVersion >= vk::ApiVersion13;

    // Check if any of the queue families support graphics operations
    auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    bool supportsGraphics =
        std::ranges::any_of(queueFamilies, [](auto const &qfp) {
          return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
        });

    // Check if all required physicalDevice extensions are available
    auto availableDeviceExtensions =
        physicalDevice.enumerateDeviceExtensionProperties();
    bool supportsAllRequiredExtensions = std::ranges::all_of(
        requiredDeviceExtension,
        [&availableDeviceExtensions](auto const &requiredExtension) {
          return std::ranges::any_of(
              availableDeviceExtensions,
              [requiredExtension](auto const &availableDeviceExtension) {
                return std::strcmp(availableDeviceExtension.extensionName,
                                   requiredExtension) == 0;
              });
        });

    // Check if the physicalDevice supports the required features
    auto features = physicalDevice.template getFeatures2<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    bool supportsRequiredFeatures =
        features.template get<vk::PhysicalDeviceVulkan11Features>()
            .shaderDrawParameters &&
        features.template get<vk::PhysicalDeviceVulkan13Features>()
            .dynamicRendering &&
        features.template get<vk::PhysicalDeviceVulkan13Features>()
            .synchronization2 &&
        features
            .template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
            .extendedDynamicState;

    // Return true if the physicalDevice meets all the criteria
    return supportsVulkan1_3 && supportsGraphics &&
           supportsAllRequiredExtensions && supportsRequiredFeatures;
  }

  void createLogicalDevice() {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
        physicalDevice.getQueueFamilyProperties();

    // get the first index into queueFamilyProperties which supports both
    // graphics and present
    queueIndex = ~0;
    for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size();
         qfpIndex++) {
      if ((queueFamilyProperties[qfpIndex].queueFlags &
           vk::QueueFlagBits::eGraphics) &&
          physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
        // found a queue family that supports both graphics and present
        queueIndex = qfpIndex;
        break;
      }
    }
    if (queueIndex == ~0) {
      throw std::runtime_error(
          "Could not find a queue for graphics and present -> terminating");
    }

    // query for Vulkan 1.3 features
    vk::StructureChain<vk::PhysicalDeviceFeatures2,
                       vk::PhysicalDeviceVulkan11Features,
                       vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
        featureChain = {
            {}, // vk::PhysicalDeviceFeatures2
            {.shaderDrawParameters =
                 true}, // vk::PhysicalDeviceVulkan11Features
            {.synchronization2 = true,
             .dynamicRendering = true}, // vk::PhysicalDeviceVulkan13Features
            {.extendedDynamicState =
                 true}, // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
        };

    // create a Device
    float queuePriority = 0.5f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
        .queueFamilyIndex = queueIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };

    vk::DeviceCreateInfo deviceCreateInfo{
        .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &deviceQueueCreateInfo,
        .enabledExtensionCount =
            static_cast<uint32_t>(requiredDeviceExtension.size()),
        .ppEnabledExtensionNames = requiredDeviceExtension.data(),
    };

    device = vk::raii::Device(physicalDevice, deviceCreateInfo);

    queue = vk::raii::Queue(device, queueIndex, 0);
  }

  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
      std::vector<vk::SurfaceFormatKHR> const &availableFormats) {
    const auto formatIt =
        std::ranges::find_if(availableFormats, [](const auto &format) {
          return format.format == vk::Format::eB8G8R8A8Srgb &&
                 format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });
    return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
  }

  vk::PresentModeKHR chooseSwapPresentMode(
      std::vector<vk::PresentModeKHR> const &availablePresentModes) {
    assert(
        std::ranges::any_of(availablePresentModes, [](const auto &presentMode) {
          return presentMode == vk::PresentModeKHR::eFifo;
        }));
    return std::ranges::any_of(availablePresentModes,
                               [](const vk::PresentModeKHR value) {
                                 return value == vk::PresentModeKHR::eMailbox;
                               })
               ? vk::PresentModeKHR::eMailbox
               : vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D
  chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities) {
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    return {std::clamp<uint32_t>(width, capabilities.minImageExtent.width,
                                 capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height,
                                 capabilities.maxImageExtent.height)};
  }

  uint32_t chooseSwapMinImageCount(
      vk::SurfaceCapabilitiesKHR const &surfaceCapabilities) {
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) &&
        (surfaceCapabilities.maxImageCount < minImageCount)) {
      minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
  }

  void createSwapChain() {
    auto surfaceCapabilities =
        physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapChainExtent = chooseSwapExtent(surfaceCapabilities);
    uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

    std::vector<vk::SurfaceFormatKHR> availableFormats =
        physicalDevice.getSurfaceFormatsKHR(*surface);
    swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);
    std::vector<vk::PresentModeKHR> availablePresentModes =
        physicalDevice.getSurfacePresentModesKHR(*surface);
    vk::PresentModeKHR presentMode =
        chooseSwapPresentMode(availablePresentModes);

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{
        .surface = *surface,
        .minImageCount = minImageCount,
        .imageFormat = swapChainSurfaceFormat.format,
        .imageColorSpace = swapChainSurfaceFormat.colorSpace,
        .imageExtent = swapChainExtent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform = surfaceCapabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = presentMode,
        .clipped = true,
        .oldSwapchain = nullptr,
    };

    swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
    swapChainImages = swapChain.getImages();
  }

  void createImageViews() {
    assert(!swapChainImages.empty());
    vk::ImageViewCreateInfo imageViewCreateInfo{
        .viewType = vk::ImageViewType::e2D,
        .format = swapChainSurfaceFormat.format,
        .subresourceRange =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };
    for (auto &image : swapChainImages) {
      imageViewCreateInfo.image = image;
      swapChainImageViews.emplace_back(device, imageViewCreateInfo);
    }
  }

  void createSyncObjects() {
    presentCompleteSemaphore =
        vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{});
    renderFinishedSemaphore =
        vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{});
    drawFence =
        vk::raii::Fence(device, vk::FenceCreateInfo{
                                    .flags = vk::FenceCreateFlagBits::eSignaled,
                                });
  }

  void cleanup() {
    glfwDestroyWindow(window);

    glfwTerminate();
  }

  [[nodiscard]] vk::raii::ShaderModule
  createShaderModule(const std::vector<char> &code) const {
    vk::ShaderModuleCreateInfo createInfo{
        .codeSize = code.size() * sizeof(char),
        .pCode = reinterpret_cast<const uint32_t *>(code.data()),
    };
    return vk::raii::ShaderModule(device, createInfo);
  }

  void createGraphicsPipeline() {
    auto shaderCode = readFile("shaders/slang.spv");
    vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shaderModule,
        .pName = "vertMain",
    };
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shaderModule,
        .pName = "fragMain",
    };
    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                        fragShaderStageInfo};

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport,
                                                   vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()};
    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList};

    vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(swapChainExtent.width),
        .height = static_cast<float>(swapChainExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vk::Rect2D scissor{
        .offset = vk::Offset2D{0, 0},
        .extent = swapChainExtent,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = vk::False,
        .lineWidth = 1.0f,
    };
    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = vk::False,
    };
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::False,
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 0,
        .pushConstantRangeCount = 0,
    };
    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::StructureChain<vk::GraphicsPipelineCreateInfo,
                       vk::PipelineRenderingCreateInfo>
        pipelineCreateInfoChain = {
            {.stageCount = 2,
             .pStages = shaderStages,
             .pVertexInputState = &vertexInputInfo,
             .pInputAssemblyState = &inputAssembly,
             .pViewportState = &viewportState,
             .pRasterizationState = &rasterizer,
             .pMultisampleState = &multisampling,
             .pColorBlendState = &colorBlending,
             .pDynamicState = &dynamicState,
             .layout = pipelineLayout,
             .renderPass = nullptr},
            {.colorAttachmentCount = 1,
             .pColorAttachmentFormats = &swapChainSurfaceFormat.format}};

    graphicsPipeline = vk::raii::Pipeline(
        device, nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }

  void createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueIndex,
    };
    commandPool = vk::raii::CommandPool(device, poolInfo);
  }

  void createCommandBuffer() {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };
    commandBuffer =
        std::move(vk::raii::CommandBuffers(device, allocInfo).front());
  }

  void transition_image_layout(uint32_t imageIndex, vk::ImageLayout old_layout,
                               vk::ImageLayout new_layout,
                               vk::AccessFlags2 src_access_mask,
                               vk::AccessFlags2 dst_access_mask,
                               vk::PipelineStageFlags2 src_stage_mask,
                               vk::PipelineStageFlags2 dst_stage_mask) {
    vk::ImageMemoryBarrier2 barrier = {
        .srcStageMask = src_stage_mask,
        .srcAccessMask = src_access_mask,
        .dstStageMask = dst_stage_mask,
        .dstAccessMask = dst_access_mask,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = swapChainImages[imageIndex],
        .subresourceRange =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };
    vk::DependencyInfo dependency_info = {
        .dependencyFlags = {},
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };
    commandBuffer.pipelineBarrier2(dependency_info);
  }

  void recordCommandBuffer(uint32_t imageIndex) {
    commandBuffer.begin({});

    // Before starting rendering, transition the swapchain image to
    // vk::ImageLayout::eColorAttachmentOptimal
    transition_image_layout(
        imageIndex, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {}, // srcAccessMask (no need to wait for previous operations)
        vk::AccessFlagBits2::eColorAttachmentWrite,         // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
        vk::PipelineStageFlagBits2::eColorAttachmentOutput  // dstStage
    );
    vk::ClearValue clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::RenderingAttachmentInfo attachInfo = {
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearValue,
    };
    vk::RenderingInfo renderingInfo = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = swapChainExtent,
            },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachInfo,
    };
    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                               *graphicsPipeline);
    commandBuffer.setViewport(
        0,
        vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
                     static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0,
                             vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();

    // After rendering, transition the swapchain image to
    // vk::ImageLayout::ePresentSrcKHR
    transition_image_layout(
        imageIndex, vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,         // srcAccessMask
        {},                                                 // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
        vk::PipelineStageFlagBits2::eBottomOfPipe           // dstStage
    );
    commandBuffer.end();
  }

  void drawFrame() {
    auto fenceResult = device.waitForFences(*drawFence, vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to wait for fence");
    }
    device.resetFences(*drawFence);
    auto [result, imageIndex] = swapChain.acquireNextImage(
        UINT64_MAX, *presentCompleteSemaphore, nullptr);

    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitDestinationStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    const vk::SubmitInfo submitInfo = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*presentCompleteSemaphore,
        .pWaitDstStageMask = &waitDestinationStageMask,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderFinishedSemaphore,
    };
    queue.submit(submitInfo, *drawFence);

    const vk::PresentInfoKHR presentInfoKHR = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinishedSemaphore,
        .swapchainCount = 1,
        .pSwapchains = &*swapChain,
        .pImageIndices = &imageIndex,
    };
    result = queue.presentKHR(presentInfoKHR);
    switch (result) {
    case vk::Result::eSuccess:
      break;
    case vk::Result::eSuboptimalKHR:
      std::cout << "vk::Queue::presentKHR returned vk::Result::eSuboptimalKHR "
                   "!\n";
      break;
    default:
      break; // an unexpected result is returned!
    }
  }
};

int main() {
  try {
    HelloTriangleApplication app;
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
